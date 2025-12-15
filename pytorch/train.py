import argparse
import os
import warnings
warnings.filterwarnings("ignore")
import torch
from torch.utils.tensorboard import SummaryWriter
import time
from copy import deepcopy
from collections import OrderedDict
import torch.optim as optim
from diffaug import DiffAugment
from utils import *
from fid import get_fid
from config import config
from image_datasets import create_loader
from model import Generator, Discriminator


@torch.no_grad()
def update_ema(ema_model, model, decay=0.999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def setup_directories(model_dir):
    # Define and create directories for logs and checkpoints
    gen_dir = os.path.join(model_dir, 'fid')
    log_img_dir = os.path.join(model_dir, 'log_img')
    log_dir = os.path.join(model_dir, 'log_dir')
    os.makedirs(gen_dir, exist_ok=True)
    os.makedirs(log_img_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    return gen_dir, log_img_dir, log_dir

def log_metrics(writer, metrics, n_images):
    for key, value in metrics.items():
        writer.add_scalar(key, value, n_images)
    writer.flush()

def evaluate_fid(ema, conf, fid_real_dir, gen_dir, 
                 n_fid_real, n_fid_gen, fid_batch_size, device):
    # Generate evaluation images and calculate FID
    gen_batches(ema, n_fid_gen, conf.gen_batch_size, conf.noise_dim, gen_dir)
    fid = get_fid(fid_real_dir, gen_dir, n_fid_real, n_fid_gen, device, fid_batch_size)
    return fid

def load_checkpoint(generator, discriminator, ema, g_opt, d_opt, model_dir):
    last_ckpt = os.path.join(model_dir, 'last_ckpt.pt')
    if os.path.exists(last_ckpt):
        ckpt = torch.load(last_ckpt)
        generator.load_state_dict(ckpt['generator'])
        discriminator.load_state_dict(ckpt['discriminator'])
        ema.load_state_dict(ckpt['ema'])
        g_opt.load_state_dict(ckpt['g_opt'])
        d_opt.load_state_dict(ckpt['d_opt'])
        start_iter = ckpt['iter'] + 1 # start from iter + 1
        best_fid = ckpt['best_fid']
        print(f'Checkpoint restored at iter {start_iter}; ' 
                f'best FID: {best_fid}')
    else:
        start_iter = 1
        best_fid = 1000. # init with big value
        print('New model initialized')
    return start_iter, best_fid

def train(model_dir, data_dir, fid_real_dir, 
          iter_interval, fid_interval, conf):
    if fid_real_dir == None:
        fid_real_dir = data_dir
    batch_size = conf.batch_size
    fid_batch_size = conf.fid_batch_size
    gen_batch_size = conf.gen_batch_size
    n_fid_real = conf.n_fid_real
    n_fid_gen = conf.n_fid_gen
    n_iter = conf.n_iter
    plot_shape = conf.plot_shape

    # dataset
    train_loader = create_loader(
        data_dir, conf.img_size, batch_size
    )
    
    # model
    generator = Generator(conf.img_size, conf.g_dim, conf.noise_dim, conf.g_heads, conf.g_mlp)
    generator.apply(weights_init)
    discriminator = Discriminator(conf.d_enc_dim, conf.d_out_dim, conf.d_heads, conf.d_mlp)
    discriminator.apply(weights_init)
    
    g_opt = optim.Adam(
        generator.parameters(), lr=conf.g_lr, betas=(conf.g_beta1, conf.g_beta2)
    )
    d_opt = optim.Adam(
        discriminator.parameters(), lr=conf.d_lr, betas=(conf.d_beta1, conf.d_beta2)
    )
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    generator.to(device)
    discriminator.to(device)
    # create ema
    ema = deepcopy(generator).to(device)  
    requires_grad(ema, False)
    
    # logs and ckpt config
    gen_dir, log_img_dir, log_dir = setup_directories(model_dir)
    writer = SummaryWriter(log_dir)
    last_ckpt = os.path.join(model_dir, './last_ckpt.pt')
    best_ckpt = os.path.join(model_dir, './best_ckpt.pt')
    
    start_iter, best_fid = load_checkpoint(
        generator, discriminator, ema, g_opt, d_opt, model_dir
    )

    # plot shape
    with torch.random.fork_rng():
        torch.manual_seed(42)  # Set your specific seed here
        seed_noise = torch.normal(
            0.0, 1.0, size=[batch_size, conf.noise_dim], device=device
        )
    # train
    start = time.time()
    metrics = {
        "g_loss_avg": 0.0,
        "d_loss_avg": 0.0,
        "d_norm_avg": 0.0,
        "g_norm_avg": 0.0,
        "fake_logits_avg": 0.0,
        "real_logits_avg": 0.0
    }

    update_ema(ema, generator, decay=0) 
    generator.train()
    discriminator.train()
    ema.eval()  # EMA model should always be in eval mode
    for idx in range(n_iter):
         # Train the generator
        i = idx + start_iter
        real_img = next(train_loader)
        real_img = real_img.to(device).requires_grad_(True)
        real_img = DiffAugment(real_img, policy=conf.policy)
        
        # Training Discriminator
        requires_grad(generator, False)
        requires_grad(discriminator, True)

        x_outputs = discriminator(real_img)
        z = torch.randn([batch_size, conf.noise_dim]).to(device)
        x_fake = generator(z).detach()
        x_fake = DiffAugment(x_fake, policy=conf.policy)
  
        z_outputs = discriminator(x_fake)
        grad_penalty = conf.r1_gamma * d_r1_loss(x_outputs, real_img) 
        d_loss = d_logistic_loss(x_outputs, z_outputs) + grad_penalty
        
        discriminator.zero_grad()
        d_loss.backward()
        d_l2_norm = torch.mean(torch.stack([torch.norm(p.grad) for p in discriminator.parameters()]))
        d_opt.step()

        # Training Generator
        requires_grad(generator, True)
        requires_grad(discriminator, False)
        z = torch.randn([batch_size, conf.noise_dim]).to(device)
        x_fake = generator(z)
        x_fake = DiffAugment(x_fake, policy=conf.policy)
        z_outputs = discriminator(x_fake)
        g_loss = g_nonsaturating_loss(z_outputs)
        
        generator.zero_grad()
        g_loss.backward()
        g_l2_norm = torch.mean(torch.stack([torch.norm(p.grad) for p in generator.parameters()]))
        g_opt.step()
        update_ema(ema, generator, decay=conf.ema_decay)
        
        # Metrics
        metrics["g_loss_avg"] += g_loss.item()
        metrics["d_loss_avg"] += d_loss.item()
        metrics["d_norm_avg"] += d_l2_norm.item()
        metrics["g_norm_avg"] += g_l2_norm.item()
        metrics["fake_logits_avg"] += z_outputs.mean().item()
        metrics["real_logits_avg"] += x_outputs.mean().item()
        
        if i % iter_interval == 0:
            # Log metrics and reset them
            n_images = i * conf.batch_size
            for key in metrics:
                metrics[key] /= iter_interval
            log_metrics(writer, metrics, n_images)
            print(f'\nTime for {n_images} images is {time.time()-start:.2f} sec '
                  f'G loss: {metrics["g_loss_avg"]:.4f} '
                  f'D loss: {metrics["d_loss_avg"]:.4f}')
            metrics = {k: 0.0 for k in metrics}  # reset metrics            
            generator.eval()
            #gen_batch = generator(seed_noise).cpu()
            gen_batch = ema(seed_noise)
            plot_path = os.path.join(log_img_dir, f'{i:04d}.png')
            plot_and_save_images(gen_batch.cpu(), plot_shape, plot_path, 
                                 img_size=conf.img_size)

            start = time.time()
            generator.train()

        if i % fid_interval == 0:
            # fid
            print('Generating eval batches...')
            fid = evaluate_fid(ema, conf, fid_real_dir, gen_dir, 
                     conf.n_fid_real, conf.n_fid_gen, fid_batch_size, device)
            print(f'FID: {fid}')
            writer.add_scalar('FID_iter', fid, i)
            writer.add_scalar('FID_n_img', fid, n_images)
            writer.flush()

            # ckpt
            ckpt_data = {
                'iter': i,
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'ema': ema.state_dict(),
                'g_opt': g_opt.state_dict(),
                'd_opt': d_opt.state_dict(),
                'fid': fid,
                'best_fid': min(fid, best_fid),
            }
            torch.save(ckpt_data, last_ckpt)
            print(f'Checkpoint saved at iter {i}')
            
            if fid <= best_fid:
                torch.save(ckpt_data, best_ckpt)
                best_fid = fid
                print(f'Best checkpoint saved at iter {i}')
                           
            start = time.time()
            generator.train()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--fid_real_dir', type=str)
    parser.add_argument('--model_dir', type=str, default='test_model')
    parser.add_argument('--iter_interval', type=int, default=1000)
    parser.add_argument('--fid_interval', type=int, default=1000)
   
    args = parser.parse_args()
    conf = Config(config, args.model_dir)

    train(
        args.model_dir, args.data_dir, args.fid_real_dir, 
        args.iter_interval, args.fid_interval, conf
    )


if __name__ == '__main__':
    main()
