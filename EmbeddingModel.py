import torch.nn as nn

class WordNetEmbeddings(nn.Module):
    def __init__(self, vocab_size, synset_size, embedding_dim):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.synset_embeddings = nn.Embedding(synset_size, embedding_dim)

    def forward(self, word_indices, synset_indices):
        word_vecs = self.word_embeddings(word_indices)
        synset_vecs = self.synset_embeddings(synset_indices)
        return word_vecs, synset_vecs