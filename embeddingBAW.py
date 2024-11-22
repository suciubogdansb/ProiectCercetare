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


def initialize_embeddings(vocab_size, embedding_dim):
    embeddings = np.random.uniform(-1, 1, (vocab_size, embedding_dim))
    return embeddings


token_embeddings = initialize_embeddings(vocab_size, EMBEDDING_DIM)


def compute_synset_embedding(synset, token_embeddings, words_to_idx):
    print(synset)
    words = synset['definition_tokens'].copy()
    words.extend(synset['examples_tokens'].copy())

    if not words:
        return np.zeros(token_embeddings.shape[1])

    token_indices = [words_to_idx[word] for word in words if word in words_to_idx]

    if not token_indices:
        return np.zeros(token_embeddings.shape[1])

    mean_embedding = np.mean(token_embeddings[token_indices], axis=0)
    weighted_embedding = mean_embedding

    return weighted_embedding


synset_embeddings = np.zeros((synset_size, EMBEDDING_DIM))

for synset in export_synset_dataset:
    idx = synset_to_idx[synset['name']]
    synset_embeddings[idx] = compute_synset_embedding(synset, token_embeddings, words_to_idx)


with open('models/synset_embeddings.npy', 'wb') as f:
    np.save(f, synset_embeddings)

