# ResNet vs Simple CNN on CIFAR-10

Comparing a simple CNN and a from-scratch ResNet implementation on CIFAR-10, including the effect of data augmentation on both architectures.

## Results

| Model | Augmentation | Parameters | Test Accuracy |
|---|---|---|---|
| SimpleCNN | No | 1,147,914 | 77.4% |
| ResNet | No | 4,907,466 | 82.1% |
| SimpleCNN | Yes | 1,147,914 | 75.5% |
| ResNet | Yes | 4,907,466 | **85.8%** |

## Architectures

**SimpleCNN** — 3 convolutional blocks (32→64→128 channels), each with BatchNorm, ReLU, and MaxPool, followed by a fully connected classifier with dropout.

**ResNet** — A custom mini-ResNet with a 3×3 prep layer followed by 4 residual blocks (64→64→128→256→512 channels). Each residual block uses two 3×3 convolutions with BatchNorm and a skip connection with projection shortcut where dimensions change.

## Observations

Augmentation (random horizontal flip, random crop, random rotation) hurt the SimpleCNN but helped the ResNet. The augmented SimpleCNN dropped from 77.4% to 75.5%, while the augmented ResNet improved from 82.1% to 85.8%.

This is a capacity effect. Data augmentation increases the effective difficulty of the dataset by introducing more variation per sample. A model needs sufficient capacity to generalise across that variation — the SimpleCNN doesn't have it, so augmentation just makes training harder without a corresponding accuracy gain. The ResNet, with ~4x more parameters and residual connections, has the capacity to benefit.

## Training

Both models trained for 20 epochs using Adam. The augmented runs use a StepLR scheduler (step size 4, gamma 0.5) in addition to augmentation.

## Requirements

```
torch
torchvision
matplotlib
scikit-learn
seaborn
numpy
```

## Usage

```bash
pip install torch torchvision matplotlib scikit-learn seaborn numpy
jupyter notebook resnet.ipynb
```