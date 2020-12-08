import re
import numpy as np
import pandas as pd
import pickle

dataset = pd.read_csv('/valohai/inputs/data/lastFiveYears.csv')
dataset = dataset[dataset['language'] == "en"]
print(dataset.shape)

from top2vec import Top2Vec

model = Top2Vec(list(dataset["txtBody_Clean"])[:100], embedding_model='universal-sentence-encoder')

with open('/valohai/outputs/file', 'wb') as f:
    pickle.dump(model, f)