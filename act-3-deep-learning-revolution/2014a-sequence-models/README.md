# Sequence Models - LSTMs & RNNs (2014)

## Images - done. What about language?

CNNs are brilliant for images - spatial patterns, filters, feature maps. But what about language? "Dog bites man" and "Man bites dog" - same words, completely different meaning. The **order** matters. CNNs don't understand order.

What about songs, stock prices, speech? These are all **sequences** - data where what came before affects what comes next. A regular neural network takes a fixed-size input and produces a fixed-size output. But sentences have different lengths, and each word depends on the words before it.

## RNN - a network with memory

The **Recurrent Neural Network** processes one word at a time and carries a **hidden state** forward - a summary of everything it's seen so far:

```
"I"     → [RNN] → hidden_1 (knows: "I")
"love"  + hidden_1 → [RNN] → hidden_2 (knows: "I love")
"cats"  + hidden_2 → [RNN] → hidden_3 (knows: "I love cats")
```

Same weights at every step - reused like a conv filter at every position. The hidden state is the memory.

The formula at each step:

```
hidden = tanh(W_input × current_word + W_hidden × previous_hidden + bias)
```

It's your Act 1 neural net - weighted sum + activation. The only new idea is feeding the output back in.

**The fatal flaw:** the hidden state passes through the same weights at every step. If the weights are small, values shrink to zero over many steps (vanishing gradient). If large, they explode. After 50 steps, the information from step 1 is gone. The RNN can't remember long-range dependencies.

## LSTM - Long Short-Term Memory

**Hochreiter and Schmidhuber (1997)** fixed this by adding a second lane - the **cell state** - where information can travel long distances without being multiplied by weights at every step.

> **Timeline note:** RNNs were proposed in 1986, LSTMs in 1997. But they became practical in **2014** when GPUs, large datasets, and the Seq2Seq architecture (which enabled machine translation) brought them to life. Same pattern as CNNs - the ideas existed for years, the impact came later.

## LSTM in depth

An LSTM unit takes three inputs:

- **Long-term memory (lm)** - the cell state
- **Short-term memory (sm)** - the hidden state
- **Current item (ci)** - the current word/character

### 1 LSTM Block - has three gates
an input word goes through these gates to complete one LSTM block.

**Forget Gate** - what to erase from long-term memory

```
f = sigmoid((wf × sm) + (wf × ci) + bf)
lm = lm × f
```

If f = 0.9, keep 90%. If f = 0.01, almost erase everything.

**Input Gate** - what new information to store in long-term memory

```
i = sigmoid((wi × sm) + (wi × ci) + bi)
candidate(potential lm) = tanh((wc × sm) + (wc × ci) + bc)
lm = lm + (candidate × i)
```

The candidate is what *could* be stored. The input gate decides how much of it actually gets stored. Notice: it's **addition** to the cell state, not multiplication. That's why information survives long distances.

**Output Gate** - what to output as the new short-term memory

```
o = sigmoid((wo × sm) + (wo × ci) + bo)
sm = o × tanh(lm)
```

Not everything in long-term memory is relevant right now. The output gate filters it.

### Key observations

- **tanh** is used to generate the *values* to add to lm or sm - **what** to add
- **sigmoid** is used to get the *percentage* of those values - **how much** to add
- In the forget gate, sigmoid decides how much to **keep**. YES, ironic name - it's really a "remember gate"
- Forget and Input gates update **long-term memory** (cell state)
- Output gate updates **short-term memory** (hidden state)
- The new sm and lm are sent to the next LSTM unit for the next item

```
inputs: current word, previous hidden state, previous cell state
  → forget gate:  what to erase from memory (long-term)
  → input gate:   what new info to store (add to long-term)
  → cell state:   updated memory (long-term)
  → output gate:  what to output right now (as short-term memory)
outputs: new hidden state, new cell state
```

## Project - Shakespeare LSTM

A **character-level language model**. Feed it Shakespeare's text, it learns to predict the next character. After training, it generates new text in Shakespeare's style.

Character-level means we don't need to handle vocabulary, tokenization, or word-level embeddings. Every character (a-z, A-Z, space, punctuation) is an input. 65 unique characters total.

### Architecture

```
Character → Embedding → LSTM (2 layers) → Fully connected → Next character prediction
```

Training data: take sequences of 100 characters. The target is the same text shifted by one character:

```
Input:    T o   b e   o r   n o
Target:   o   b e   o r   n o t
```

At every position, the model predicts the next character. After training, generate text by predicting one character at a time and feeding it back in.

## New concepts

### Vector Embedding

Characters (or words) need to become numbers for the network. But assigning 'a' = 0, 'b' = 1 creates false relationships - the network thinks 'b' is more similar to 'a' than to 'z'.

Instead, each character gets a **vector** - a list of 128 numbers. These start random and are learned during training. Characters that behave similarly in the text end up with similar vectors.

`nn.Embedding(65, 128)` is a lookup table: 65 rows (one per character), 128 columns (the vector size). Feed in character index 3, get back row 3.

### Model persistence

```python
torch.save(model.state_dict(), "model.pth")   # save trained weights
model.load_state_dict(torch.load("model.pth")) # load them later
```

Train once, save, reuse forever. This is what happens behind the scenes when you download a pretrained model - someone trained it, saved the weights, and you load them.

### PyTorch - how Python won

The original framework was called **Torch**, written for the **Lua** programming language. It was powerful but Lua had a small community. In 2016, Facebook AI Research rewrote it in Python and called it **PyTorch**. Python's massive ecosystem - NumPy, pandas, Scikit-learn, Jupyter notebooks - made it the natural home for ML. Researchers switched almost overnight. Google had **TensorFlow** (2015), but PyTorch's simpler, more Pythonic design won the research community. By 2019, the majority of ML papers used PyTorch.

The framework wars mirror the architecture wars - the most elegant, developer-friendly tool wins, not necessarily the most powerful one.

---

## Short Notes

**Previous problem:** Neural networks could handle images (CNNs) but couldn't process sequences - language, time series, speech. Order matters, and CNNs don't understand order.

**Solution this provided:** RNNs introduced memory through a hidden state. LSTMs fixed the vanishing gradient problem with a protected cell state and three gates. Machines could now read, translate, and generate text - laying the foundation for transformers and eventually GPT.
