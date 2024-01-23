import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from tqdm import tqdm
import numpy as np
from PIL import Image
import math
import json
import tensorflow as tf
from tensorflow.keras import layers


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
    
def save_generator_heads(model, epoch, noise, main_dir, 
			resolution_dirs, heads, size=15, n_resolutions=3):
    predictions, maps = model(noise, training=False)
    predictions = np.clip(deprocess(predictions), 0, 255).astype(np.uint8)

    fig = plt.figure(figsize=(size, size))

    for i in range(predictions.shape[0]):
        # create subplot and append to ax
        fig.add_subplot(8, 8, i+1)
        plt.imshow(predictions[i, :, :, :])
        plt.axis('off')

    path = os.path.join(main_dir, f'{epoch:04d}.png')
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    plt.savefig(path, format='png')
    plt.close()
    
    for r in range(n_resolutions):
        for h in range(heads[r]):
            map_size = int(math.sqrt(maps[r][0][0].shape[0])) # get map high and width 
            fig = plt.figure(figsize=(size, size))
            for i in range(predictions.shape[0]):
                fig.add_subplot(8, 8, i+1)
                map_reshape = tf.reshape(maps[r][i][h], [map_size, map_size])
                plt.imshow(map_reshape)
                plt.axis('off')
            plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

            path = os.path.join(resolution_dirs[r], f'ep{epoch:04d}_r{r}_h{str(h)}.png')
            plt.savefig(path)
            plt.close()

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
    def __init__(self, input_dict, save_dir):
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

        # Update the object's attributes with loaded configuration
        for key, value in config_data.items():
            setattr(self, key, value)
        print(f'Config {file_path} loaded')