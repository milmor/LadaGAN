import os
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from utils import deprocess
import math


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

def get_map(maps, resolution, head):
    h = head
    r = resolution
    map_size = int(math.sqrt(maps[r][0][0].shape[0])) # get map high and width 
    b = maps[r][:, h].shape[0]
    reshaped_maps = tf.reshape(maps[r][:, h], [b, map_size, map_size, 1])
    return reshaped_maps

def plot_single_head(predictions, maps, h=1, size=2):
    n = len(predictions)
    maps32 = get_map(maps, 2, h)
    maps16 = get_map(maps, 1, h)
    maps8 = get_map(maps, 0, h)
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(4, n, figsize=(n*size, 4*size))

    # Plot the images in the first row
    for i in range(n):
        axes[0, i].imshow(predictions[i])
        axes[0, i].axis('off')

    # Plot maps32 in the second row
    for i in range(n):
        axes[1, i].imshow(maps32[i, :, :, 0])
        axes[1, i].axis('off')

    # Plot maps16 in the third row
    for i in range(n):
        axes[2, i].imshow(maps16[i, :, :, 0])
        axes[2, i].axis('off')

    # Plot maps8 in the fourth row
    for i in range(n):
        axes[3, i].imshow(maps8[i, :, :, 0])
        axes[3, i].axis('off')

    # Remove the titles
    for ax in axes.ravel():
        ax.set_title("")

    # Display the plot
    plt.tight_layout()
    plt.show()