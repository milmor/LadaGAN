"""LadaGAN model for Tensorflow.

Reference:
  - [Efficient generative adversarial networks using linear 
    additive-attention Transformers](https://arxiv.org/abs/2401.09596)
"""
import tensorflow as tf
import os
from tqdm import tqdm
from PIL import Image
import time
from diffaug import DiffAugment
from utils import deprocess
from fid import *


def l2_loss(y_true, y_pred):
    return tf.reduce_mean(
        tf.keras.losses.mean_squared_error(y_true, y_pred)
    )

def reset_metrics(metrics):
    for _, metric in metrics.items():
        metric.reset_states()

def update_metrics(metrics, **kwargs):
    for metric_name, metric_value in kwargs.items():
        metrics[metric_name].update_state(metric_value)

        
class LadaGAN(object):
    def __init__(self, generator, discriminator, conf):
        super(LadaGAN, self).__init__()
        self.generator = generator
        self.ema_generator = generator
        self.discriminator = discriminator
        self.noise_dim = conf.noise_dim
        self.gp_weight = conf.gp_weight
        self.policy = conf.policy
        self.batch_size = conf.batch_size
        self.ema_decay = conf.ema_decay
        self.ema_generator = tf.keras.models.clone_model(generator)
        # init ema
        noise = tf.random.normal([1, conf.noise_dim])
        gen_batch = self.ema_generator(noise)

        # metrics
        self.train_metrics = {}
        self.fid_avg = tf.keras.metrics.Mean()

    def build(self, g_optimizer, d_optimizer, g_loss, d_loss):
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.g_loss = g_loss
        self.d_loss = d_loss
        self._build_metrics()

    def _build_metrics(self):
        metric_names = [
        'g_loss',
        'd_loss',
        'gp',
        'd_total',
        'real_acc',
        'fake_acc'
        ]
        for metric_name in metric_names:
            self.train_metrics[metric_name] = tf.keras.metrics.Mean()

    def gradient_penalty(self, real_samples):
        batch_size = tf.shape(real_samples)[0]
        with tf.GradientTape() as gradient_tape:
            gradient_tape.watch(real_samples)
            logits = self.discriminator(real_samples, training=True)[0]

        r1_grads = gradient_tape.gradient(logits, real_samples)
        r1_grads = tf.reshape(r1_grads, (batch_size, -1))
        r1_penalty = tf.reduce_sum(tf.square(r1_grads), axis=-1)
        r1_penalty = tf.reduce_mean(r1_penalty) * self.gp_weight 
        return logits, r1_penalty

    @tf.function(jit_compile=True)
    def train_step(self, real_images):

        noise = tf.random.normal(shape=[self.batch_size, self.noise_dim])
        # train the discriminator
        with tf.GradientTape() as d_tape:
            fake_images = self.generator(noise, training=True)[0]
            fake_augmented_images = DiffAugment(fake_images, policy=self.policy) 
            real_augmented_images = DiffAugment(real_images, policy=self.policy)
            fake_logits = self.discriminator(fake_augmented_images, training=True)[0]
            real_logits, gp = self.gradient_penalty(real_augmented_images)          

            d_loss = self.d_loss(real_logits, fake_logits)
            d_total = d_loss + gp 

        d_gradients = d_tape.gradient(d_total, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(d_gradients, self.discriminator.trainable_weights)
        )

        noise = tf.random.normal(shape=[self.batch_size, self.noise_dim])
        # train the generator 
        with tf.GradientTape() as g_tape:
            fake_images = self.generator(noise, training=True)[0]
            fake_augmented_images = DiffAugment(fake_images, policy=self.policy) 
            fake_logits = self.discriminator(fake_augmented_images, training=True)[0]
            g_loss = self.g_loss(fake_logits)
            
        g_gradients = g_tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(
            zip(g_gradients, self.generator.trainable_weights)
        )
        for weight, ema_weight in zip(self.generator.weights, self.ema_generator.weights):
            ema_weight.assign(self.ema_decay * ema_weight + (1 - self.ema_decay) * weight)
            
        update_metrics(
         self.train_metrics,
         g_loss=g_loss,
         d_loss=d_loss,
         gp=gp,
         d_total=d_total,
         real_acc=tf.reduce_mean(real_logits),
         fake_acc=tf.reduce_mean(fake_logits)   
      )
        
    def create_ckpt(self, model_dir, max_ckpt_to_keep, restore_best=True):
        # log dir
        self.model_dir = model_dir
        log_dir = os.path.join(model_dir, 'log-dir')
        self.writer = tf.summary.create_file_writer(log_dir)
        
        # checkpoint dir
        checkpoint_dir = os.path.join(model_dir, 'training-checkpoints')
        best_checkpoint_dir = os.path.join(
            model_dir, 'best-training-checkpoints'
        )

        self.ckpt = tf.train.Checkpoint(g_optimizer=self.g_optimizer,
                d_optimizer=self.d_optimizer, generator=self.generator,
                ema_generator=self.ema_generator, 
                discriminator=self.discriminator,
                n_images=tf.Variable(0),
                fid=tf.Variable(10000.0), # initialize with big value
                best_fid=tf.Variable(10000.0), # initialize with big value
        ) 
        
        self.ckpt_manager = tf.train.CheckpointManager(
            self.ckpt, directory=checkpoint_dir, 
            max_to_keep=max_ckpt_to_keep
         )
        self.best_ckpt_manager = tf.train.CheckpointManager(
            self.ckpt, directory=best_checkpoint_dir, 
            max_to_keep=max_ckpt_to_keep
        )
               
        if restore_best == True and self.best_ckpt_manager.latest_checkpoint:    
            last_ckpt = self.best_ckpt_manager.latest_checkpoint
            self.ckpt.restore(last_ckpt)
            print(f'Best checkpoint restored from {last_ckpt}')
        elif restore_best == False and self.ckpt_manager.latest_checkpoint:
            last_ckpt = self.ckpt_manager.latest_checkpoint
            self.ckpt.restore(last_ckpt)
            print(f'Checkpoint restored from {last_ckpt}')     
        else:
            print(f'Checkpoint created at {model_dir} dir')
            
    def restore_generator(self, model_dir):
        self.model_dir = model_dir
        log_dir = os.path.join(model_dir, 'log-dir')
        
        # checkpoint dir
        best_checkpoint_dir = os.path.join(
            model_dir, 'best-training-checkpoints'
        )

        self.ckpt = tf.train.Checkpoint(
                ema_generator=self.ema_generator, 
                n_images=tf.Variable(0),
        ) 

        self.best_ckpt_manager = tf.train.CheckpointManager(
            self.ckpt, directory=best_checkpoint_dir, 
            max_to_keep=max_ckpt_to_keep
        )
                  
        last_ckpt = self.best_ckpt_manager.latest_checkpoint
        self.ckpt.restore(last_ckpt)
        print(f'Best checkpoint restored from {last_ckpt}')

    def save_metrics(self, n_images): 
        # tensorboard  
        with self.writer.as_default():
            for name, metric in self.train_metrics.items():
                print(f'{name}: {metric.result():.4f} -', end=" ")
                tf.summary.scalar(name, metric.result(), step=n_images)
         # reset metrics        
        reset_metrics(self.train_metrics)
            
    def save_ckpt(self, n_images, n_fid_images, fid_batch_size, test_dir, img_size):
        # fid
        fid = self.fid(n_fid_images, fid_batch_size, test_dir, img_size)
        self.fid_avg.update_state(fid)
        with self.writer.as_default():
            tf.summary.scalar('FID_n_img', self.fid_avg.result(), step=n_images)
            
        # checkpoint
        self.ckpt.n_images.assign(n_images)
        self.ckpt.fid.assign(fid)       
        
        start = time.time()
        if fid < self.ckpt.best_fid:
            self.ckpt.best_fid.assign(fid)
            self.best_ckpt_manager.save(n_images)
            self.ckpt_manager.save(n_images)
            print(f'FID improved. Best checkpoint saved at {n_images} images') 
        else:
            self.ckpt_manager.save(n_images)
            print(f'Checkpoint saved at {n_images} images')  
        print(f'Time for ckpt is {time.time()-start:.4f} sec') 
        
        # reset metrics
        self.fid_avg.reset_states()   
        
    def gen_batches(self, n_images, batch_size, dir_path):
        n_batches = n_images // batch_size
        for i in tqdm(range(n_batches)):
            start = i * batch_size
            noise = tf.random.normal([batch_size, self.noise_dim])
            gen_batch = self.ema_generator(noise, training=False)[0]
            gen_batch = np.clip(deprocess(gen_batch), 0.0, 255)

            img_index = start
            for img in gen_batch:
                img = Image.fromarray(img.astype('uint8'))
                file_name = os.path.join(dir_path, f'{str(img_index)}.png')
                img.save(file_name,"PNG")
                img_index += 1
                
    def fid(self, n_fid_images, batch_size, test_dir, img_size):
        inception = Inception()
        fid_dir = os.path.join(self.model_dir, 'fid')
        os.makedirs(fid_dir, exist_ok=True)
        # fid
        start = time.time()
        print('\nGenerating FID images...') 
        self.gen_batches(n_fid_images, batch_size, fid_dir)
        gen_fid_ds = create_fid_ds(
            fid_dir + '/*.png', batch_size, img_size, n_fid_images
        )
        real_fid_ds = create_fid_ds(
            test_dir, batch_size, img_size, n_fid_images
        )
        m_gen, s_gen = calculate_activation_statistics(
            gen_fid_ds, inception, batch_size
        )
        m_real, s_real = calculate_activation_statistics(
            real_fid_ds, inception, batch_size
        )        
        fid = calculate_frechet_distance(m_real, s_real, m_gen, s_gen)
        print(f'FID: {fid:.4f} - Time for FID score is {time.time()-start:.4f} sec')            
        return fid 