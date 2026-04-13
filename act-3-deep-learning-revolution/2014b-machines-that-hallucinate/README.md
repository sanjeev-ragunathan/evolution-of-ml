# Machines That Hallucinate — GANs (2014)

## Everything so far has been about understanding. What about creating?

Every model we've built takes input and classifies it — is this a cat? Did this passenger survive? What digit is this? The model *understands* but it doesn't *create*.

In 2014, **Ian Goodfellow** had an idea at a bar in Montreal. What if you set up two networks in a **competition**?

## The adversarial game

**The Generator** — starts with random noise (just a list of random numbers) and tries to create a fake image. It has never seen a real image. It's guessing blindly.

**The Discriminator** — looks at images and tries to tell real from fake. It's seen real images from the dataset.

They train together, pushing each other:

```
Generator creates a fake image
  → Discriminator says "fake!"
    → Generator learns from that feedback, gets slightly better
      → Discriminator has to get better at detecting
        → Generator improves again
          → ...
```

The generator gets better at faking. The discriminator gets better at detecting. Eventually, the generator produces images so realistic that the discriminator can't tell the difference.

Goodfellow went home that night and coded the whole thing. **It worked on the first try.** He named it **GAN** — Generative Adversarial Network.

## GAN vs DCGAN

The original GAN (2014) used fully connected layers — flatten the image to numbers, process through linear layers. It worked, but results were blurry. Fully connected layers don't understand spatial structure.

**DCGAN** (2015, Radford et al.) replaced them with **convolutional layers** — the same spatial awareness from LeNet and AlexNet. The discriminator uses convolutions to detect patterns. The generator uses **transposed convolutions** to build patterns up from noise.

```
GAN (2014):   fully connected layers → blurry results
DCGAN (2015): convolutional layers   → sharp, realistic images
```

## The architecture

**Discriminator** — a CNN that outputs one number between 0 (fake) and 1 (real):

```
Image (28×28)
  → Conv (32 filters, 3×3, stride=2) → ReLU     → 13×13
  → Conv (64 filters, 3×3, stride=2) → ReLU     → 6×6
  → Flatten (2304)
  → Linear → Sigmoid → one number (0 to 1)
```

This is basically LeNet but with a single sigmoid output instead of 10 classes.

**Generator** — the reverse. Takes noise and *upscales* it into an image:

```
Random noise (64 numbers)
  → Linear → Reshape to (128, 7, 7)
  → ConvTranspose2d (128→64, 4×4, stride=2, padding=1) → ReLU    → 14×14
  → ConvTranspose2d (64→1, 4×4, stride=2, padding=1) → Tanh      → 28×28
```

`ConvTranspose2d` is the opposite of `Conv2d` — where convolution shrinks, transposed convolution grows. 7→14→28.

The final activation is **tanh** (output between -1 and 1) to match how we normalize the MNIST images.

## The training loop — two models, one game

This is different from everything before. You're training **two models simultaneously** with **two optimizers**:

```
For each batch:
  1. Train Discriminator:
     - Real images → should predict 1 (real)
     - Fake images → should predict 0 (fake)
     - Update discriminator weights
  
  2. Train Generator:
     - Generate fake images
     - Pass through discriminator
     - Loss against target 1 (fool the discriminator)
     - Update generator weights
```

### Why target = 1 when training the generator?

The generator's goal is to **fool** the discriminator. It wants the discriminator to say 1 (real) for its fake images. The loss measures how far from that goal it is:

- Discriminator says 0.1 for a fake → high loss → generator changes a lot
- Discriminator says 0.8 for a fake → low loss → generator is already fooling it

We're not telling the generator fake images are real. We're telling it: **"your job is to make the discriminator output 1."**

```
Training discriminator:
  real images → target 1 (correctly identify real)
  fake images → target 0 (correctly identify fake)

Training generator:
  fake images → target 1 (fool the discriminator)
```

The discriminator learns to tell truth. The generator learns to lie convincingly.

### The detach trick

```python
# Training discriminator — detach stops gradients flowing to generator
fake_output = discriminator(fake_images.detach())

# Training generator — no detach, gradients flow to generator
fake_output = discriminator(fake_images)
```

When training the discriminator, we don't want the generator to change. `.detach()` breaks the gradient chain. When training the generator, we need gradients to flow through the discriminator all the way back to the generator's weights.

## The balance problem

What if the discriminator becomes too good? It catches every fake with 100% confidence. The gradient becomes useless — "you're wrong" with no direction for improvement. The generator stagnates.

What if the generator becomes too good too fast? The discriminator is fooled every time and can't learn either.

They need to stay **roughly balanced**, improving at similar rates. This is why GANs are notoriously tricky to train — the loss doesn't just go down smoothly like your other models. It oscillates as the two networks compete.

## New concepts

### BCELoss — Binary Cross Entropy

The loss function for binary (real/fake) predictions. It measures how wrong a probability is:

```
prediction = 0.9, target = 1  → loss = 0.105  (confident and right — small loss)
prediction = 0.1, target = 1  → loss = 2.302  (confident and wrong — huge loss)
```

Being confidently wrong is punished much more than being slightly wrong. Designed specifically for sigmoid outputs between 0 and 1.

### Transposed Convolution

The reverse of convolution. Where `Conv2d` shrinks images, `ConvTranspose2d` grows them. The generator uses this to upscale from a tiny 7×7 feature map to a full 28×28 image.

---

## Short Notes

**Previous problem:** Neural networks could classify and understand, but couldn't create. All models so far take input and produce a label — none of them generate new data.

**Solution this provided:** GANs introduced adversarial training — two networks competing, one creating, one judging. The generator learns to produce realistic images from pure noise. This is the foundation of all generative AI — DALL-E, Stable Diffusion, and Midjourney all descend from this idea.