'''
LSTM
- Generates Shakespeare style text.
- Character level language model. So we don't have to handle vocab, tokenization and embeddings.
'''

# ruff: noqa: E402 # to ignore "imports not on top of the file" warning

import torch
import torch.nn as nn

class LSTM_Shakespeare(nn.Module):

    # ARCHITECTURE
    # vocab size - no.of unique chars in the dataset
    # embed size - the size of the vector used to represent a char
    # hidden_size - the size of the hidden state vector in the LSTM layer.
    # num_layers - number of LSTM layers we want to stack
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x): # x - batch of sequences.
        embed_out = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embed_out) # returns output, hidden state and cell state after full completion.
        fc_out = self.fc(lstm_out) # FC layer outputs predictions for every position simultaneously, and the loss function compares all of them at once.
        return fc_out



# DATASET

# LOAD DATASET
import urllib.request
import os

if not os.path.exists("./data/shakespeare.txt"):
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    urllib.request.urlretrieve(url, "./data/shakespeare.txt")

with open("./data/shakespeare.txt", "r") as f:
    text = f.read()

print(f"Total characters: {len(text)}")
print(f"First 200 characters:\n{text[:200]}")

# CREATE VOCAB
chars = sorted(set(text))
vocab_size = len(chars)
char_to_idx = {char: idx for idx, char in enumerate(chars)}
idx_to_char = {idx: char for idx, char in enumerate(chars)}

print(f"Vocab Size: {vocab_size}")
print(f"Characters: {''.join(chars)}")

# TEXT TO NUMBERS
data = [char_to_idx[char] for char in text]
data = torch.tensor(data, dtype=torch.long)

# CREATE SEQUENCES
# The idea: take chunks of text, and for each chunk the target is the same text shifted by one character.

def create_sequences(data, seq_length):
    inputs = []
    targets = []
    for i in range(len(data) - seq_length):
        inputs.append(data[i:i+seq_length])
        targets.append(data[i+1:i+seq_length+1])
    return torch.stack(inputs), torch.stack(targets)

seq_length = 100
X, Y = create_sequences(data[:50000], seq_length)
print(f"Input shape: {X.shape}, Target shape: {Y.shape}")



# TRAIN

from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

dataset = TensorDataset(X, Y)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

model = LSTM_Shakespeare(vocab_size, embed_size=128, hidden_size=256, num_layers=2)

# loading saved model if exists, else train and save the model
model_path = "shakespeare_model.pth"
if os.path.exists(model_path):
    print("Loading saved model...")
    model.load_state_dict(torch.load(model_path))
else:
    print("Training model...")
    # your training loop here

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    for epoch in range(10):
        for batch_X, batch_Y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = loss_fn(outputs.view(-1, vocab_size), batch_Y.view(-1))
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), model_path)



# TEXT GENERATION

def generate(model, start_char, length=500):
    model.eval()
    input = torch.tensor([[char_to_idx[start_char]]])
    result = [start_char]
    
    with torch.no_grad():
        for _ in range(length):
            output = model(input)
            # Get probabilities for the last character
            probs = torch.softmax(output[0, -1], dim=0)
            # Sample from the distribution (not just picking the highest)
            next_idx = torch.multinomial(probs, 1).item()
            result.append(idx_to_char[next_idx])
            # Feed the predicted character back in
            input = torch.tensor([[next_idx]])
    
    return ''.join(result)

print(generate(model, 'T'))