import argparse
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Disable tensorflow debugging logs
import time
import tensorflow as tf
import json
from tqdm import tqdm
from PIL import Image
from model import Generator, Discriminator
from utils import *
from trainer import LadaGAN
from config import config


def train(file_pattern, eval_dir, model_dir, metrics_inter, 
          fid_inter, total_iter, max_ckpt_to_keep, conf):
    train_ds = create_train_iter_ds(file_pattern, conf.batch_size, conf.img_size)

    # init model
    noise = tf.random.normal([conf.batch_size, conf.noise_dim])
    generator = Generator(model_dim=conf.g_dim, heads=conf.g_heads, 
                          mlp_dim=conf.g_mlp)
    gen_batch = generator(noise)
    generator.summary()
    print('G output shape:', gen_batch[0].shape)
    
    discriminator = Discriminator(model_dim=conf.d_dim,                              
                                  heads=conf.d_heads,
                                  mlp_dim=conf.d_mlp,
                                  initializer=conf.d_initializer)
    out_disc = discriminator(
        tf.ones([conf.batch_size, conf.img_size, conf.img_size, 3])
    )
    discriminator.summary()
    print('D Output shape:', out_disc[0].shape)
    
    gan = LadaGAN(generator=generator, discriminator=discriminator, 
                  conf=conf)
    
    # define losses
    generator_loss, discriminator_loss = get_loss(conf.loss)

    gan.build(
        g_optimizer=tf.keras.optimizers.Adam(
            learning_rate=conf.g_lr, 
            beta_1=conf.g_beta1, 
            beta_2=conf.g_beta2
        ),
        d_optimizer=tf.keras.optimizers.Adam(
            learning_rate=conf.d_lr, 
            beta_1=conf.d_beta1,
            beta_2=conf.d_beta2),
        g_loss=generator_loss,
        d_loss=discriminator_loss)

    gan.create_ckpt(model_dir, max_ckpt_to_keep, restore_best=False)
    
    # train config
    num_examples_to_generate = 64 # plot images
    noise_seed = tf.random.normal([num_examples_to_generate, 
                                   conf.noise_dim], seed=conf.test_seed)
    gen_img_dir = os.path.join(model_dir, 'log-gen-img')
    os.makedirs(gen_img_dir, exist_ok=True)
    # additive attention maps dir for 3 stages
    n_resolutions = 3
    resolution_dirs = []
    for resolution in range(n_resolutions):
        path = os.path.join(gen_img_dir, 'res_{}'.format(resolution))
        os.makedirs(path, exist_ok=True)                    
        resolution_dirs.append(path)
        
    start_iter = int((gan.ckpt.n_images / gan.batch_size) + 1)
    
    # train
    n_images = int(gan.ckpt.n_images)
    start = time.time()
    for idx in range(start_iter, total_iter):
        image_batch = train_ds.get_next()
        gan.train_step(image_batch)

        if idx % metrics_inter == 0:
            print(f'\nTime for metrics_inter is {time.time()-start:.4f} sec')
            n_images = idx * gan.batch_size
            gan.save_metrics(n_images)
            start = time.time()

        if idx % fid_inter == 0:
            n_images = idx * gan.batch_size
            gan.save_ckpt(n_images, conf.n_fid_real, 
                conf.fid_batch_size, eval_dir, conf.img_size
            )
            save_generator_heads(
                gan.ema_generator, n_images, noise_seed, 
                gen_img_dir, resolution_dirs, conf.g_heads, size=conf.plot_size
            )
            start = time.time()

            
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_pattern', type=str)
    parser.add_argument('--eval_dir', type=str)
    parser.add_argument('--model_dir', type=str, default='model_1')
    parser.add_argument('--metrics_inter', type=int, default=500)
    parser.add_argument('--fid_inter', type=int, default=500)
    parser.add_argument('--total_iter', type=int, default=10000000000)
    parser.add_argument('--max_ckpt_to_keep', type=int, default=2)
    args = parser.parse_args()

    conf = Config(config, args.model_dir)
    train(
        args.file_pattern, args.eval_dir, args.model_dir, 
        args.metrics_inter, args.fid_inter, args.total_iter,
        args.max_ckpt_to_keep, conf
    )


if __name__ == '__main__':
    main()