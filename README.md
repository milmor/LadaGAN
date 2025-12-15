# LadaGAN
This repo is the official implementation of "[Efficient generative adversarial networks using linear additive-attention Transformers](https://arxiv.org/abs/2401.09596)".

<img src="./images/ffhq_128_img.png" width="850px"></img>

By Emilio Morales-Juarez and Gibran Fuentes-Pineda.


## Abstract
> Although the capacity of deep generative models for image generation, such as Diffusion Models (DMs) and Generative Adversarial Networks (GANs), has dramatically improved in recent years, much of their success can be attributed to computationally expensive architectures. This has limited their adoption and use to research laboratories and companies with large resources, while significantly raising the carbon footprint for training, fine-tuning, and inference. In this work, we present a novel GAN architecture which we call LadaGAN. This architecture is based on a linear attention Transformer block named Ladaformer. The main component of this block is a linear additive-attention mechanism that computes a single attention vector per head instead of the quadratic dot-product attention. We employ Ladaformer in both the generator and discriminator, which reduces the computational complexity and overcomes the training instabilities often associated with Transformer GANs. LadaGAN consistently outperforms existing convolutional and Transformer GANs on benchmark datasets at different resolutions while being significantly more efficient. Moreover, LadaGAN shows competitive performance compared to state-of-the-art multi-step generative models (e.g. DMs) using orders of magnitude less computational resources.

## Implementations

This repository provides **two implementations** of LadaGAN:

- **[TensorFlow Implementation](./tensorflow/)** - Original TensorFlow implementation
- **[PyTorch Implementation](./pytorch/)** - PyTorch implementation

Both implementations are feature-complete and produce equivalent results. Choose the one that best fits your workflow.

### Quick Start

**TensorFlow:**
```bash
cd tensorflow
python train.py --file_pattern=./data_path/*png --eval_dir=./eval_path/*png
```

**PyTorch:**
```bash
cd pytorch
python train.py --data_dir ./data_path --fid_real_dir ./eval_path
```

For detailed instructions, dependencies, and usage, please refer to the README files in each implementation directory:
- [TensorFlow README](./tensorflow/README.md)
- [PyTorch README](./pytorch/README.md)


## FLOPs
Training on CIFAR-10 and CelebA using a single 12GB GPU (RTX 3080 Ti) takes less than 40 hours. __Note that these results and the experiments reported in the paper were obtained using the [TensorFlow implementation of LadaGAN](./tensorflow/README.md), which runs twice as fast as the PyTorch implementation due to XLA.__  
| Model (CIFAR 10 32x32) | ADM-IP (80 steps) | StyleGAN2 |  VITGAN  | LadaGAN  |
| :-- |  :------:  |  :------:  |  :------:   |  :------:  |
| GPUs | Tesla V100 x 2| - |- | __RTX 3080 Ti x 1__ |
|   #Images | 69M | - | - | __68M__ |
| #Params | 57M | - | - | __19M__ |
| FLOPs | 9.0B | - | - | __0.7B__ |
| FID | __2.93__| 5.79 |4.57 | 3.29 |

| Model (CelebA 64x64)  | ADM-IP (80 steps) | StyleGAN2 |  VITGAN  | LadaGAN  |
| :-- |  :------:  |  :------:  |  :------:   |  :------:  |
| GPUs | Tesla V100 x 16| - | - | __RTX 3080 Ti x 1__ |
|   #Images | 138M |- |- | __72M__ |
| #Params | 295M | 24M | 38M | __19M__ |
| FLOPs | 103.5B | 7.8B |2.6B | __0.7B__ |
| FID | 2.67| -|3.74 | __1.81__ |

| Model (FFHQ 128x128)  | ADM-IP (80 steps) | StyleGAN2 |  VITGAN  | LadaGAN  |
| :-- |  :------:  |  :------:  |  :------:   |  :------:  |
|   #Images  | 61M | - |  - | __53M__ |
| #Params | 543M | - | - | __24M__ |
| FLOPs | 391.0B| 11.5B |11.8B| __4.3B__ |
| FID| 6.89| - | - | __4.48__ |


## Hyperparameters
Adjust hyperparameters in the `config.py` file within each implementation directory.

## Implementation Notes
- This model depends on other files that may be licensed under different open source licenses.
- LadaGAN uses [Differentiable Augmentation](https://arxiv.org/abs/2006.10738). Under BSD 2-Clause "Simplified" License.
- [FID](https://arxiv.org/abs/1706.08500) evaluation.
- TensorFlow implementation: Efficient patch generation with XLA.


## Demo
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ZS7pSxh_-PLSFAcJwuG0WCejD5cRTg9C?)


## Attention maps
Single head maps training progress:

<img src="./images/learning_bedroom128.gif" width="600px"></img>
<img src="./images/learning_ffhq128.gif" width="600px"></img>

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

