# The Big Bang Moment (2012)

## CNNs worked... but only on easy problems

LeNet could read handwritten digits at 95%+ accuracy. But MNIST was easy - 28×28 grayscale images, 10 categories, clean centered digits. The moment you tried real-world photos with thousands of categories, cluttered backgrounds, and varying lighting - it fell apart. Not because the idea was wrong, but because the hardware and data weren't ready.

## 1998–2012: The quiet winter

Nobody killed neural nets this time. People just stopped caring.

- **Compute wasn't there.** Training a CNN on real images took weeks or months on the CPUs of the late 90s. LeCun's LeNet worked on tiny images. Scaling it up was impractical.
- **Data wasn't there.** Deep networks need massive amounts of labeled data. In 1998, there was no ImageNet. No internet-scale photo collections. Gathering and labeling millions of images was a research project in itself.
- **Classical ML was "good enough."** SVMs, random forests, and ensemble methods were winning competitions. Faster to train, easier to understand, solid theory. Why gamble on neural nets?
- **Academic culture turned against neural nets.** If you submitted a neural network paper to a top conference in 2005, reviewers would often reject it. The field had decided that kernel methods and graphical models were the future. Hinton, LeCun, and a few others kept working - Hinton later called their group the "deep learning conspiracy."

## 2006-2012: Three things changed

- **2006 - Hinton's deep belief networks.** Hinton published a paper showing you could pre-train deep networks layer by layer. First proof that *deep* networks (not just 2-3 layers) could work. The deep learning community started to re-form.
- **2009 - ImageNet.** **Fei-Fei Li's** team releases a dataset of 14 million labeled images across 20,000 categories. The annual ImageNet Large Scale Visual Recognition Challenge (ILSVRC) starts in 2010 — 1.2 million images, 1000 categories. For the first time, there's enough data to train something big.
- **2007-2012 - GPUs.** NVIDIA had been making graphics cards for video games. Rendering 3D graphics requires massively parallel matrix multiplication — exactly what neural networks need. Training that took months on CPU took days on GPU.

## 2012: AlexNet

**Alex Krizhevsky**, a graduate student at the University of Toronto under **Geoffrey Hinton**, combined all three - deep architecture, massive data, GPU training - and entered the ImageNet competition.

The result:

```
Best classical ML entry:  26.2% error
AlexNet:                  15.3% error
```

A **10.8 percentage point gap**. In a competition where teams fought over fractions of a percent. AlexNet didn't just win — it obliterated the field with a neural network, the thing everyone had written off.

The modern AI boom traces directly back to this moment.

## What made AlexNet different

Same idea as LeNet - conv layers, pooling, fully connected. But four critical innovations:

### ReLU — fixing what sigmoid and tanh couldn't

- **Sigmoid** (1986): squashes output between 0 and 1. Maximum derivative is **0.25**. In an 8-layer network: `0.25^8 = 0.00001526`. The gradient reaching the first layer is essentially zero. Early layers stop learning.
- **Tanh** (1998): squashes between -1 and 1. Derivative maxes at **1.0**, but still vanishes at the extremes where the curve flattens.
- **ReLU** (2012): `max(0, x)`. Derivative is **1** for all positive values. No shrinking, no vanishing. Multiply 1 through 8 layers and you still get 1. The gradient arrives at full strength. ReLU isn't mathematically fancy — it's `max(0, x)`. But it unlocked depth.

### Dropout — solving co-dependency and overfitting

**The problem: overfitting.** A network with millions of parameters can memorize training data instead of learning general patterns. It gets 99% accuracy on training images but fails on new ones. It learned "that specific photo of a cat" instead of "what cats look like."

**The deeper problem: co-dependency.** Without dropout, neurons get lazy. One neuron learns "pointy ears" perfectly. Others stop trying and just check if the ear neuron fired. They depend on it completely. Show the network a cat with folded ears? The ear neuron doesn't fire, and all the lazy neurons that depended on it have nothing to work with. Everything breaks — even though there are whiskers, fur, eye shape, and a hundred other cat features it *could* have learned.

**The solution: dropout.** Randomly **turn off 50% of neurons** during each training step. Different neurons every time. Each neuron is forced to learn something useful on its own — it can't rely on any partner always being there.

It's like a group project where you randomly remove team members each day. Everyone is forced to learn every part of the project because they can't count on anyone else being there tomorrow.

During testing, all neurons are active. The full network works, but each neuron is robust because it was trained under pressure.

### Data augmentation - preventing memorization

60 million parameters can easily memorize 1.2 million training images. Solution: transform each image randomly every time the network sees it.

- Flip horizontally - a cat facing left is still a cat
- Crop slightly differently - the cat doesn't need to be centered
- Shift colors - a cat in dim lighting is still a cat

The network never sees the exact same image twice. It's forced to learn "cat" instead of memorizing specific photos.

```python
transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
])
```

### GPU training

AlexNet had **60 million parameters** (LeNet had ~44,000). Training this on CPU with 1.2 million images would take months. Krizhevsky split the network across two NVIDIA GTX 580 GPUs and trained it in about a week.

## Building AlexNet on CIFAR-10

We can't train on ImageNet — it's 150GB and would take hours even on a GPU. Instead we use **CIFAR-10**: 60,000 images across 10 categories (airplane, car, bird, cat, deer, dog, frog, horse, ship, truck). Small enough for a MacBook, complex enough to feel the difference from MNIST.

### Adapting the architecture

Original AlexNet uses 11×11 filters with stride 4 — designed for 227×227 images. On CIFAR-10's 32×32 images, an 11×11 filter would cover a third of the image in one step. So we keep the same depth and spirit but shrink filter sizes:

```
Original AlexNet (227×227)                    Our AlexNet (32×32)
Conv1: 96,  11×11, stride 4 → ReLU → Pool    Conv1: 96,  3×3 → ReLU → MaxPool(2)
Conv2: 256, 5×5 → ReLU → Pool                Conv2: 256, 3×3 → ReLU → MaxPool(2)
Conv3: 384, 3×3 → ReLU                       Conv3: 384, 3×3, padding=1 → ReLU
Conv4: 384, 3×3 → ReLU                       Conv4: 384, 3×3, padding=1 → ReLU
Conv5: 256, 3×3 → ReLU → Pool                Conv5: 256, 3×3, padding=1 → ReLU → MaxPool(2)
Dropout → FC1: 4096 → ReLU                   Dropout → FC1: 4096 → ReLU
Dropout → FC2: 4096 → ReLU                   Dropout → FC2: 4096 → ReLU
FC3: 1000 classes                             FC3: 10 classes
```

**Padding** — `padding=1` adds a 1-pixel border of zeros around the image before convolving, so the feature map stays the same size instead of shrinking. Without it, after 5 conv layers the feature maps would disappear.

### Data normalization

Neural networks train better when inputs are centered around 0. CIFAR-10 pixels are 0–1 after `ToTensor()`. Normalizing shifts them to -1 to 1:

```python
transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
```

Same concept as `StandardScaler` in the Titanic chapter — keeping inputs in a balanced range. Krizhevsky did this too — he subtracted the mean pixel value from every image.

## The first run failed

First attempt: plain SGD (no momentum), no data normalization. The result:

```
Epoch 0: Loss 2.30
Epoch 1: Loss 2.30
Epoch 2: Loss 2.30
Epoch 3: Loss 2.29
Epoch 4: Loss 2.17
Test Accuracy: 0.1466
```

14.66% — barely better than random guessing (10% for 10 classes). The loss was flat for 4 epochs. The network wasn't learning.

**Diagnosis:** two problems.

**1. No momentum.** Plain SGD looks only at the current batch's gradient. With **momentum** (`momentum=0.9`), the optimizer remembers the direction it's been moving — like a ball rolling downhill that builds up speed instead of stopping and restarting at every step. This is what the original AlexNet used.

**2. No normalization.** Pixel values weren't centered. The network struggled to find good initial gradients.

After adding momentum and normalization:

```
Epoch 0: Loss 1.98
Epoch 1: Loss 1.53
Epoch 2: Loss 1.22
Epoch 3: Loss 0.61
Epoch 4: Loss 1.20
Test Accuracy: 0.6658
```

14.66% → **66.58%**. The loss is actually dropping. The network is learning.

This is real ML — models rarely work first try. Debugging is the job.

## The result

```
LeNet (1998) on MNIST:      95.3%  - 5 epochs, ~10 seconds
AlexNet (2012) on CIFAR-10: 66.6%  - 5 epochs, ~16 minutes
```

Different dataset difficulty (clean digits vs real photos), but the difference in scale is what matters. LeNet: 44,000 parameters, seconds to train. AlexNet: millions of parameters, minutes to train on a MacBook. On the original ImageNet, Krizhevsky needed two GPUs and a full week.

## Key Insight

AlexNet proved that neural networks weren't a dead end - they just needed scale. The algorithms existed since the 1980s. What was missing was compute (GPUs), data (ImageNet), and a few clever tricks (ReLU, dropout, data augmentation). When all three arrived at the same time, deep learning exploded.