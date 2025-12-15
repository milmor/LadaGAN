import torch
from torch import nn
import torch.nn.functional as F
from torch import autograd
from torchvision.utils import save_image
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from tqdm import tqdm

def denormalize(image):  
    return (image + 1.) / 2.

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
        self.print_variables()

    def load_config(self, file_path):
        # Load configuration from the existing file
        with open(file_path, "r") as f:
            config_data = json.load(f)

        # Update the object's attributes with loaded configuration
        for key, value in config_data.items():
            setattr(self, key, value)
        print(f'Config {file_path} loaded')
        self.print_variables()
        
    def print_variables(self):
        # Print all variables (attributes) of the Config object
        for key, value in vars(self).items():
            print(f"{key}: {value}")

def plot_and_save_images(batch, plot_shape, filename, img_size=32):
    # Convert the batch tensor to a numpy array
    batch = batch.numpy()
    
    # Get the number of images to plot based on plot_shape
    rows, cols = plot_shape
    num_images = rows * cols

    # Ensure that the batch size is at least as large as the number of images to plot
    if batch.shape[0] < num_images:
        raise ValueError(f"Batch size is smaller than the number of images to plot. "
                         f"Batch size: {batch.shape[0]}, required: {num_images}.")

    # Create a figure with subplots
    fig, axes = plt.subplots(rows, cols, figsize=(cols * img_size / 90, rows * img_size / 90))

    # Flatten axes for easy indexing
    axes = axes.flatten()

    for i in range(num_images):
        # Get the image (channel 0 for grayscale or channel 1 for color images)
        image = batch[i][0] if batch.shape[1] == 1 else batch[i].transpose(1, 2, 0)
        
        # Rescale the image to [0, 1] range if necessary
        if image.min() < 0 or image.max() > 1:
            image = (image - image.min()) / (image.max() - image.min())
        
        # Plot the image
        axes[i].imshow(image)
        axes[i].axis('off')  # Hide axis

    # Adjust layout to prevent overlapping
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    # Save the image
    plt.savefig(filename)
    plt.close()

def gen_batches(model, n_images, batch_size, noise_dim, dir_path):
    n_batches = n_images // batch_size
    device = next(model.parameters()).device
    for i in tqdm(range(n_batches)):
        start = i * batch_size
        #noise = torch.randn(batch_size, noise_dim, device=device)
        noise = torch.randn([batch_size, noise_dim], device=device)
        gen_batch = model(noise)
        gen_batch = gen_batch.cpu()
        gen_batch = denormalize(gen_batch)
        gen_batch = gen_batch.clamp(min=0, max=1)

        img_index = start
        for img in gen_batch:
            file_name = os.path.join(dir_path, f'{str(img_index)}.png')
            save_image(img, file_name) # expect range [0, 1]
            img_index += 1

def weights_init(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.orthogonal_(model.weight.data)
    if classname.find('Linear') != -1:
        nn.init.orthogonal_(model.weight.data)
    if classname.find('Parameter') != -1:
        nn.init.orthogonal_(model.weight.data)

def d_logistic_loss(real_pred, fake_pred):
    assert type(real_pred) == type(fake_pred)
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)
    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()
    return loss