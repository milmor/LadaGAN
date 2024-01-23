# LadaGAN
This repo is the official implementation of "[Efficient generative adversarial networks using linear additive-attention Transformers](https://arxiv.org/abs/2401.09596)".

By Emilio Morales-Juarez and Gibran Fuentes-Pineda


## Abstract
> Although the capacity of deep generative models for image generation, such as Diffusion Models (DMs) and Generative Adversarial Networks (GANs), has dramatically improved in recent years, much of their success can be attributed to computationally expensive architectures. This has limited their adoption and use to research laboratories and companies with large resources, while significantly raising the carbon footprint for training, fine-tuning, and inference. In this work, we present LadaGAN, an efficient generative adversarial network that is built upon a novel Transformer block named Ladaformer. The main component of this block is a linear additive-attention mechanism that computes a single attention vector per head instead of the quadratic dot-product attention. We employ Ladaformer in both the generator and discriminator, which reduces the computational complexity and overcomes the training instabilities often associated with Transformer GANs. LadaGAN consistently outperforms existing convolutional and Transformer GANs on benchmark datasets at different resolutions while being significantly more efficient. Moreover, LadaGAN shows competitive performance compared to state-of-the-art multi-step generative models (e.g. DMs) using orders of magnitude less computational resources. 


## BibTeX
```bibtex
@article{morales2024efficient,
  title={Efficient generative adversarial networks using linear additive-attention Transformers},
  author={Morales-Juarez, Emilio and Fuentes-Pineda, Gibran},
  journal={arXiv preprint arXiv:2401.09596},
  year={2024}
}
