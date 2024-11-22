import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle

from EmbeddingModel import WordNetEmbeddings

print("load data...")
with open('data/synset_dataset.pkl', 'rb') as f:
    export_synset_dataset = pickle.load(f)

with open('data/words_to_idx.pkl', 'rb') as f:
    words_to_idx = pickle.load(f)

with open('data/synset_to_idx.pkl', 'rb') as f:
    synset_to_idx = pickle.load(f)

with open('data/idx_to_synset.pkl', 'rb') as f:
    idx_to_synset = pickle.load(f)

with open('data/idx_to_words.pkl', 'rb') as f:
    idx_to_words = pickle.load(f)

with open('data/other.json', 'rb') as f:
    other = pickle.load(f)
    vocab_size = other["vocab_size"]
    synset_size = other["synset_size"]
    EMBEDDING_DIM = other["EMBEDDING_DIM"]
    CATEGORY_WEIGHTS = other["CATEGORY_WEIGHTS"]

print("init model...")



EPOCHS = 250
LEARNING_RATE = 0.075
OPTIMIZER = "sgd"

model = WordNetEmbeddings(vocab_size, synset_size, EMBEDDING_DIM)
criterion = nn.CosineEmbeddingLoss()

if OPTIMIZER == "adam":
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
elif OPTIMIZER == "sgd":
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
else:
    raise ValueError("Invalid optimizer")

print("train model...")


def train_model(model, data, epochs):
    for epoch in range(epochs):
        total_loss = 0
        for entry in data:
            gloss_tokens = entry["definition_tokens"]
            example_tokens = entry["examples_tokens"]
            all_tokens = gloss_tokens + example_tokens

            word_indices = torch.tensor([words_to_idx[word] for word in all_tokens if word in words_to_idx])
            if word_indices.shape[0] == 0:
                continue

            synset_index = torch.tensor([synset_to_idx[entry["name"]]])

            category = entry["pos"]
            weight = CATEGORY_WEIGHTS.get(category, 1.0)

            word_vecs, synset_vecs = model(word_indices, synset_index)
            target = torch.ones(word_vecs.shape[0])
            loss = criterion(word_vecs, synset_vecs.expand_as(word_vecs), target) * weight

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss}")


train_model(model, export_synset_dataset, EPOCHS)

torch.save(model.state_dict(), f'models/model{OPTIMIZER}{LEARNING_RATE}{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.pth')


