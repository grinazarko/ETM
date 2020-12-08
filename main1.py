import re
import numpy as np
import pandas as pd
import pickle
import pip
import importlib

def install(package):
    if hasattr(pip, 'main'):
        pip.main(['install', package])
    else:
        pip._internal.main(['install', package])

install("top2vec")
install("top2vec[sentence_encoders]")

dataset = pd.read_csv('/valohai/inputs/data/lastFiveYears.csv')
dataset = dataset[dataset['language'] == "en"]
print(dataset.shape)

import top2vec 

importlib.reload(top2vec)

model = top2vec.Top2Vec(list(dataset["txtBody_Clean"])[:100], embedding_model='universal-sentence-encoder')

with open('/valohai/outputs/file', 'wb') as f:
    pickle.dump(model, f)