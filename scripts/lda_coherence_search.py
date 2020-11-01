import pickle
import gensim
import os
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.coherencemodel import CoherenceModel


with open(os.path.join("/valohai/inputs/data", 'vectorizer.pkl'), 'rb') as f:
    vectorizer, train_set_vectorized, test_set_vectorized = pickle.load(f)

topics_numbers = [5, 10, 20, 50, 70, 100, 300, 500, 700, 1000, 3000]
topic_coherences = []
models=[]

print("start")

def get_coherence_values(topics_numbers_parameter):
    corpus_vect_gensim_train = gensim.matutils.Sparse2Corpus(train_set_vectorized, documents_columns=False)
    id_map = dict([(i, s) for i, s in enumerate(vectorizer.get_feature_names())])
    lda = LdaMulticore(corpus=corpus_vect_gensim_train, id2word=id_map, num_topics=topics_numbers_parameter, workers=3)
    models.append(lda)
    print(topics_numbers_parameter)
    corpus_vect_gensim_test = gensim.matutils.Sparse2Corpus(test_set_vectorized, documents_columns=False)    
    cm = CoherenceModel(model=lda, corpus=corpus_vect_gensim_test, coherence='u_mass', processes=-1)
    coherence = cm.get_coherence()
    topic_coherences.append(coherence)
    print(coherence, topics_numbers_parameter)
    
for topic_number in topics_numbers:
    get_coherence_values(topic_number)

print(topic_coherences)   