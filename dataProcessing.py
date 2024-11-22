import json

from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords as sw
from nltk.corpus import wordnet2022 as wn22
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import pickle

WORDS_PER_POS = 300
EMBEDDING_DIM = 35

synset_dataset_partial = []

print('Downloading NLTK resources...')


def get_synset_data(pos):
    for i, synset in enumerate(wn22.all_synsets(pos=pos)):
        if i == WORDS_PER_POS:
            break
        data = {
            "name": synset.name(),
            "lemmas": synset.lemma_names(),
            "definition": synset.definition(),
            'pos': synset.pos(),
            "examples": synset.examples(),
        }
        synset_dataset_partial.append(data)


poss = ["a", "n", "v", "r", "s"]
print('Extracting synset data...')
for p in poss:
    get_synset_data(p)

lema = WordNetLemmatizer()
stop_words = set(sw.words('english'))


def process_text(text):
    tokens = word_tokenize(text)
    tokens = [lema.lemmatize(token) for token in tokens if token.isalpha() and token not in stop_words]
    return tokens


print('Processing synset data...')
vocabulary = set()
for synset in synset_dataset_partial:
    definition_tokens = process_text(synset['definition'])
    vocabulary.update(definition_tokens)
    examples_tokens = []
    for example in synset['examples']:
        example_tokens = process_text(example)
        examples_tokens.extend(example_tokens)
    vocabulary.update(synset['lemmas'])
    vocabulary.update(examples_tokens)
    synset['definition_tokens'] = definition_tokens
    synset['examples_tokens'] = examples_tokens

print('Creating vocabulary...')
words_to_idx = {word: idx for idx, word in enumerate(vocabulary)}
idx_to_words = {idx: word for word, idx in words_to_idx.items()}

synset_to_idx = {synset['name']: idx for idx, synset in enumerate(synset_dataset_partial)}
idx_to_synset = {idx: synset for synset, idx in synset_to_idx.items()}

vocab_size = len(vocabulary)
synset_size = len(synset_dataset_partial)
export_synset_dataset = synset_dataset_partial

CATEGORY_WEIGHTS = {
    "n": 1.0,
    "v": 0.85,
    "a": 0.8,
    "r": 0.65,
    "s": 0.8
}

saveJson = {
    "CATEGORY_WEIGHTS": CATEGORY_WEIGHTS,
    "vocab_size": vocab_size,
    "synset_size": synset_size,
    "EMBEDDING_DIM": EMBEDDING_DIM
}


print('Saving processed data...')
with open('data/synset_dataset.pkl', 'wb') as f:
    pickle.dump(export_synset_dataset, f)

with open('data/words_to_idx.pkl', 'wb') as f:
    pickle.dump(words_to_idx, f)

with open('data/synset_to_idx.pkl', 'wb') as f:
    pickle.dump(synset_to_idx, f)

with open('data/idx_to_synset.pkl', 'wb') as f:
    pickle.dump(idx_to_synset, f)

with open('data/idx_to_words.pkl', 'wb') as f:
    pickle.dump(idx_to_words, f)

with open('data/other.json', 'wb') as f:
    pickle.dump(saveJson, f)

print('Data processing complete!')
