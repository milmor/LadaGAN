## LadaGAN &mdash; Official PyTorch implementation

<img src="./images/plot_ffhq.png" width="850px"></img>

This repository is a reimplementation of [LadaGAN](https://github.com/milmor/LadaGAN) in PyTorch.

> [**Efficient generative adversarial networks using linear additive-attention Transformers**](https://arxiv.org/abs/2401.09596)<br>
> By Emilio Morales-Juarez and Gibran Fuentes-Pineda.


## Abstract
> Although the capacity of deep generative models for image generation, such as Diffusion Models (DMs) and Generative Adversarial Networks (GANs), has dramatically improved in recent years, much of their success can be attributed to computationally expensive architectures. This has limited their adoption and use to research laboratories and companies with large resources, while significantly raising the carbon footprint for training, fine-tuning, and inference. In this work, we present a novel GAN architecture which we call LadaGAN. This architecture is based on a linear attention Transformer block named Ladaformer. The main component of this block is a linear additive-attention mechanism that computes a single attention vector per head instead of the quadratic dot-product attention. We employ Ladaformer in both the generator and discriminator, which reduces the computational complexity and overcomes the training instabilities often associated with Transformer GANs. LadaGAN consistently outperforms existing convolutional and Transformer GANs on benchmark datasets at different resolutions while being significantly more efficient. Moreover, LadaGAN shows competitive performance compared to state-of-the-art multi-step generative models (e.g. DMs) using orders of magnitude less computational resources.


## Training LadaGAN 
Use `--data_dir=<data_dir>` and `--fid_real_dir=<fid_real_dir>` to specify the dataset path and the FID evaluation path.  
```bash
python train.py --data_dir='../datasets/celeba_64_train/' --fid_real_dir='../datasets/celeba_64_train/'
```  

## FLOPs  
Training on CIFAR-10 and CelebA using a single 12GB GPU (RTX 3080 Ti) takes less than 40 hours. __Note that these results and the experiments reported in the paper were obtained using the [TensorFlow implementation of LadaGAN](https://github.com/milmor/LadaGAN), which runs twice as fast as the PyTorch implementation due to XLA.__  
| Model (CIFAR 10 32x32) | ADM-IP (80 steps) | StyleGAN2 |  VITGAN  | LadaGAN  |
| :-- |  :------:  |  :------:  |  :------:   |  :------:  |
| GPUs | Tesla V100 x 2| - |- | __RTX 3080 Ti x 1__ |
|   #Images | 69M |- |- | __68M__ |
| #Params | 57M | - |- | __19M__ |
| FLOPs | 9.0B | - | - | __0.7B__ |
| FID | __2.93__| 5.79 |4.57 | 3.29 |

| Model (CelebA 64x64)  | ADM-IP (80 steps) | StyleGAN2 |  VITGAN  | LadaGAN  |
| :-- |  :------:  |  :------:  |  :------:   |  :------:  |
| GPUs | Tesla V100 x 16| - |- | __RTX 3080 Ti x 1__ |
|   #Images | 138M |- |- | __72M__ |
| #Params | 295M | 24M | 38M | __19M__ |
| FLOPs | 103.5B | 7.8B |2.6B | __0.7B__ |
| FID | 2.67| -|3.74 | __1.81__ |

| Model (FFHQ 128x128)  | ADM-IP (80 steps) | StyleGAN2 |  VITGAN  | LadaGAN  |
| :-- |  :------:  |  :------:  |  :------:   |  :------:  |
|   #Images  | 61M | - |  - | __53M__ |
| #Params | 543M | - | - | __24M__ |
| FLOPs | 391.0B| 11.5B |11.8B| __4.3B__ |
| FID| 6.89| - | -| __4.48__ |


## Hparams setting
Adjust hyperparameters in the `config.py` file.

Implementation notes:
- This model depends on other files that may be licensed under different open source licenses.
- LadaGAN uses [Differentiable Augmentation](https://arxiv.org/abs/2006.10738). Under BSD 2-Clause "Simplified" License.
- [FID](https://arxiv.org/abs/1706.08500) evaluation.
- Currently, the model only supports patch generation.  
- ⚠️ Warning: Due to subtle differences between TensorFlow 2 and PyTorch, we modified the R1 coefficient; adjust it if changing resolution.
## To-Do  
- Add bCR  
- Add generator conv decoder 


## BibTeX
```bibtex
@article{morales2024efficient,
  title={Efficient generative adversarial networks using linear additive-attention Transformers},
  author={Morales-Juarez, Emilio and Fuentes-Pineda, Gibran},
  journal={arXiv preprint arXiv:2401.09596},
  year={2024}
}
```


## License
MIT