# Conditional Generative Adversarial Network (cGAN) for MNIST

This repository contains an implementation of a Conditional Generative Adversarial Network (cGAN) for generating MNIST digits conditioned on the label.

## Quick Start

### 1. Setup the environment:

You can create a conda environment with all the necessary dependencies using:

```bash
conda env create -f conda_env.yml
```

After the environment is set up, activate it:

```bash
conda activate cgan_env
```
### 2. Train the model:

Navigate to the main directory and start the training:

```bash
python train.py
```

## Features

- Modular and well-structured code for easy understanding and customization.
- Uses PyTorch for model definition, training, and inference.
- Training script with command line arguments for hyperparameters.
- Loading/Saving checkpoints functionality.
- Option to visualize tensor sizes for debugging purposes.

## Future work:

- Implement evaluation metrics for model assessment.
- Extend to other datasets.







