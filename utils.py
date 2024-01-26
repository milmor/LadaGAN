import os
import json
import tensorflow as tf
from tensorflow.keras import layers
from huggingface_hub import hf_hub_download
import json


AUTOTUNE = tf.data.experimental.AUTOTUNE


def deprocess(img):
    return img * 127.5 + 127.5

def train_convert(file_path, img_size):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [img_size, img_size])
    img = (img - 127.5) / 127.5 
    return img

def create_train_iter_ds(train_dir, batch_size, img_size):
    img_paths = tf.data.Dataset.list_files(str(train_dir))
    BUFFER_SIZE = tf.data.experimental.cardinality(img_paths)

    img_paths = img_paths.cache().shuffle(BUFFER_SIZE)
    ds = img_paths.map(lambda img: train_convert(img, img_size), 
            num_parallel_calls=AUTOTUNE)

    ds = ds.batch(batch_size, drop_remainder=True, 
            num_parallel_calls=AUTOTUNE)
    print(f'Train dataset size: {BUFFER_SIZE}')
    print(f'Train batches: {tf.data.experimental.cardinality(ds)}')
    ds = ds.repeat().prefetch(AUTOTUNE)
    return iter(ds)

def get_loss(loss):
    if loss == 'nsl':
        def discriminator_loss(real_img, fake_img):
            real_loss = tf.reduce_mean(tf.math.softplus(-real_img))
            fake_loss = tf.reduce_mean(tf.math.softplus(fake_img)) 
            return real_loss + fake_loss

        def generator_loss(fake_img):
            return tf.reduce_mean(tf.math.softplus(-fake_img))

        return generator_loss, discriminator_loss

    elif loss == 'hinge':
        def d_real_loss(logits):
            return tf.reduce_mean(tf.nn.relu(1.0 - logits))

        def d_fake_loss(logits):
            return tf.reduce_mean(tf.nn.relu(1.0 + logits))

        def discriminator_loss(real_img, fake_img):
            real_loss = d_real_loss(real_img)
            fake_loss = d_fake_loss(fake_img)
            return fake_loss + real_loss

        def generator_loss(fake_img):
            return -tf.reduce_mean(fake_img)

        return generator_loss, discriminator_loss

class Config(object):
    def __init__(self, save_dir, input_dict=None):
        if input_dict != None:
            for key, value in input_dict.items():
                setattr(self, key, value)
        file_path = os.path.join(save_dir, "config.json")

        # Check if the configuration file exists
        if os.path.exists(file_path):
            self.load_config(file_path)
        else:
            self.save_config(file_path, save_dir)

    def save_config(self, file_path, save_dir):
        # Create the directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        # Convert input_dict to JSON and save to file
        with open(file_path, "w") as f:
            json.dump(vars(self), f, indent=4)
        print(f'New config {file_path} saved')

    def load_config(self, file_path):
        # Load configuration from the existing file
        with open(file_path, "r") as f:
            config_data = json.load(f)

        print(f'Config {file_path} loaded')
        # Update the object's attributes with loaded configuration
        for key, value in config_data.items():
            print(f'{key}: {value}')
            setattr(self, key, value)
        
        
class Loader(object):
    def __init__(self):
        pass
        
    def download(self, ckpt_dir):
        repo_id = 'milmor/LadaGAN'
        if ckpt_dir == 'ffhq_128':
            hf_hub_download(repo_id=repo_id, 
                filename=f"{ckpt_dir}/best-training-checkpoints/ckpt-24064000.data-00000-of-00001",
                local_dir='./'
            )

            hf_hub_download(repo_id=repo_id, 
                filename=f"{ckpt_dir}/best-training-checkpoints/ckpt-24064000.index",
                local_dir='./'
            )

            hf_hub_download(repo_id=repo_id, 
                filename=f"{ckpt_dir}/best-training-checkpoints/checkpoint",
                local_dir='./')

            config_file = hf_hub_download(repo_id=repo_id, 
                filename=f"{ckpt_dir}/config.json",
                local_dir='./'
            )
        elif ckpt_dir == 'celeba_64':
            hf_hub_download(repo_id=repo_id, 
                filename=f"{ckpt_dir}/best-training-checkpoints/ckpt-72192000.data-00000-of-00001",
                local_dir='./'
            )

            hf_hub_download(repo_id=repo_id, 
                filename=f"{ckpt_dir}/best-training-checkpoints/ckpt-72192000.index",
                local_dir='./'
            )

            hf_hub_download(repo_id=repo_id, 
                filename=f"{ckpt_dir}/best-training-checkpoints/checkpoint",
                local_dir='./')

            config_file = hf_hub_download(repo_id=repo_id, 
                filename=f"{ckpt_dir}/config.json",
                local_dir='./'
            )
        elif ckpt_dir == 'cifar10':
            hf_hub_download(repo_id=repo_id, 
                filename=f"{ckpt_dir}/best-training-checkpoints/ckpt-68096000.data-00000-of-00001",
                local_dir='./'
            )

            hf_hub_download(repo_id=repo_id, 
                filename=f"{ckpt_dir}/best-training-checkpoints/ckpt-68096000.index",
                local_dir='./'
            )

            hf_hub_download(repo_id=repo_id, 
                filename=f"{ckpt_dir}/best-training-checkpoints/checkpoint",
                local_dir='./')

            config_file = hf_hub_download(repo_id=repo_id, 
                filename=f"{ckpt_dir}/config.json",
                local_dir='./'
            )

        with open(config_file) as f:
            self.config = json.load(f)
