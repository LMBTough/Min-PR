# Towards Minimising Perturbation Rate for Adversarial Machine Learning with Pruning

This repository contains the code and instructions to reproduce the experiments in the paper "Towards Minimising Perturbation Rate for
Adversarial Machine Learning with Pruning".

## Requirements

To run this code, you will need to have the following Python packages installed:

- foolbox==3.3.3
- numpy==1.24.2
- pandas==1.5.3
- Pillow==9.5.0
- torch==2.0.0+cu118
- torchvision==0.15.1+cu118
- tqdm==4.65.0

## Downloading the Necessary Files

You will also need to download the following files:
- Weights: https://drive.google.com/file/d/1_RCpxcbJHJ5y3V4iCoO8Y0cp_vOeUeGv/view?usp=sharing
- Data: https://drive.google.com/file/d/1v9mvo5JzVOS7erOhEUQq7GorCVA4e9FI/view?usp=sharing

## Generating Adversarial Examples

Before running the experiment, you need to generate adversarial examples by running the `generate_adv.py` script. Here is an example command:
```bash
    python generate_adv.py --dataset cifar10 --attack linfpgd
```

## Running the Experiment

After generating the adversarial examples, you can run the main experiment using the `main.py` script. Here is an example command:

```bash
    python main.py --dataset cifar10 --method total --attack linfpgd --pic_num 1000 --alpha 0.01 --use_label
```
The above command will run the experiment using 1000 images from the CIFAR-10 dataset and the LinfPGD attack with a step size of 0.01. The `--use_label` flag is use label before attack.

Please note that running the above commands may take some time, depending on the specifications of your machine.
