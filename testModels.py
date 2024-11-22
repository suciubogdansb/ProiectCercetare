import os
import pickle
import numpy as np

import torch
from nltk.corpus import wordnet2022 as wn22

from EmbeddingModel import WordNetEmbeddings

person = wn22.synset('person.n.01')
animal = wn22.synset('animal.n.01')
plant = wn22.synset('plant.n.02')
organism = wn22.synset('organism.n.01')

shopping = wn22.synset('shopping.n.01')
buying = wn22.synset('buying.n.01')
marketing = wn22.synset('marketing.n.03')

with open('data/other.json', 'rb') as f:
    other = pickle.load(f)
    vocab_size = other["vocab_size"]
    synset_size = other["synset_size"]
    EMBEDDING_DIM = other["EMBEDDING_DIM"]

with open('data/synset_to_idx.pkl', 'rb') as f:
    synset_to_idx = pickle.load(f)

with open('data/words_to_idx.pkl', 'rb') as f:
    words_to_idx = pickle.load(f)

personIdx = synset_to_idx['person.n.01']
animalIdx = synset_to_idx['animal.n.01']
plantIdx = synset_to_idx['plant.n.02']
organismIdx = synset_to_idx['organism.n.01']

shoppingIdx = synset_to_idx['shopping.n.01']
buyingIdx = synset_to_idx['buying.n.01']
marketingIdx = synset_to_idx['marketing.n.03']

for filename in os.listdir('models'):
    if not filename.endswith('.pth'):
        continue
    model = WordNetEmbeddings(vocab_size, synset_size, EMBEDDING_DIM)  # Define model architecture
    model.load_state_dict(torch.load(f"models/{filename}"))
    model.eval()

    with torch.no_grad():
        person_synset = torch.tensor([personIdx])
        animal_synset = torch.tensor([animalIdx])
        plant_synset = torch.tensor([plantIdx])
        organism_synset = torch.tensor([organismIdx])

        shopping_synset = torch.tensor([shoppingIdx])
        buying_synset = torch.tensor([buyingIdx])
        marketing_synset = torch.tensor([marketingIdx])

        person_vec = model.synset_embeddings(person_synset)
        animal_vec = model.synset_embeddings(animal_synset)
        plant_vec = model.synset_embeddings(plant_synset)
        organism_vec = model.synset_embeddings(organism_synset)

        shopping_vec = model.synset_embeddings(shopping_synset)
        buying_vec = model.synset_embeddings(buying_synset)
        marketing_vec = model.synset_embeddings(marketing_synset)

        person_animal = torch.cosine_similarity(person_vec, animal_vec).item()
        person_plant = torch.cosine_similarity(person_vec, plant_vec).item()
        person_organism = torch.cosine_similarity(person_vec, organism_vec).item()

        shopping_buying = torch.cosine_similarity(shopping_vec, buying_vec).item()
        shopping_marketing = torch.cosine_similarity(shopping_vec, marketing_vec).item()

        plant_marketing = torch.cosine_similarity(plant_vec, marketing_vec).item()

        person_word_vec = model.word_embeddings(torch.tensor([words_to_idx['person']]))

        person_person = torch.cosine_similarity(person_vec, person_word_vec).item()
        print(f"Model: {filename}")
        print(f"person_animal: {person_animal}")
        print(f"person_plant: {person_plant}")
        print(f"person_organism: {person_organism}")
        print(f"shopping_buying: {shopping_buying}")
        print(f"shopping_marketing: {shopping_marketing}")
        print(f"person_marketing: {plant_marketing}")
        print(f"person_person: {person_person}")
        print("\n")


print("NLTK WordNet Similarity Wu-Palmer")
print(f"person_animal: {person.wup_similarity(animal)}")
print(f"person_plant: {person.wup_similarity(plant)}")
print(f"person_organism: {person.wup_similarity(organism)}")
print(f"shopping_buying: {shopping.wup_similarity(buying)}")
print(f"shopping_marketing: {shopping.wup_similarity(marketing)}")
print("\n")
print("NLTK WordNet Similarity Path")
print(f"person_animal: {person.path_similarity(animal)}")
print(f"person_plant: {person.path_similarity(plant)}")
print(f"person_organism: {person.path_similarity(organism)}")
print(f"shopping_buying: {shopping.path_similarity(buying)}")
print(f"shopping_marketing: {shopping.path_similarity(marketing)}")
print("\n")
print("NLTK WordNet Similarity Leacock-Chodorow")
print(f"person_animal: {person.lch_similarity(animal)}")
print(f"person_plant: {person.lch_similarity(plant)}")
print(f"person_organism: {person.lch_similarity(organism)}")
print(f"shopping_buying: {shopping.lch_similarity(buying)}")
print(f"shopping_marketing: {shopping.lch_similarity(marketing)}")

with open('models/synset_embeddings.npy', 'rb') as f:
    synset_embeddings = np.load(f)

    person_embedding = synset_embeddings[personIdx]
    animal_embedding = synset_embeddings[animalIdx]
    plant_embedding = synset_embeddings[plantIdx]
    organism_embedding = synset_embeddings[organismIdx]

    shopping_embedding = synset_embeddings[shoppingIdx]
    buying_embedding = synset_embeddings[buyingIdx]
    marketing_embedding = synset_embeddings[marketingIdx]

    person_animal = np.dot(person_embedding, animal_embedding) / (np.linalg.norm(person_embedding) * np.linalg.norm(animal_embedding))
    person_plant = np.dot(person_embedding, plant_embedding) / (np.linalg.norm(person_embedding) * np.linalg.norm(plant_embedding))
    person_organism = np.dot(person_embedding, organism_embedding) / (np.linalg.norm(person_embedding) * np.linalg.norm(organism_embedding))

    shopping_buying = np.dot(shopping_embedding, buying_embedding) / (np.linalg.norm(shopping_embedding) * np.linalg.norm(buying_embedding))
    shopping_marketing = np.dot(shopping_embedding, marketing_embedding) / (np.linalg.norm(shopping_embedding) * np.linalg.norm(marketing_embedding))

    plant_marketing = np.dot(plant_embedding, marketing_embedding) / (np.linalg.norm(plant_embedding) * np.linalg.norm(marketing_embedding))
    print("BAW, non NN model")
    print(f"person_animal: {person_animal}")
    print(f"person_plant: {person_plant}")
    print(f"person_organism: {person_organism}")
    print(f"shopping_buying: {shopping_buying}")
    print(f"shopping_marketing: {shopping_marketing}")
    print(f"plant_marketing: {plant_marketing}")
    print("\n")

