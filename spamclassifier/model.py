import pickle

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

from spamclassifier.preprocess import clean_text#, custom_prepro

def train(X,y,save=True):
    prepro = FunctionTransformer(lambda x : x['text'].apply(clean_text))
    #prepro = custom_prepro()
    pipe = Pipeline([
        ('preprocess',prepro),
        ('tf_idf',TfidfVectorizer(ngram_range = (1,2), max_features = 200)),
        ('model',MultinomialNB(alpha=0.1,class_prior=(0.5,0.5)))
    ])

    pipe.fit(X,y)
    if save:
        filename = "../model/classifier.pickle"
        pickle.dump(pipe, open(filename, "wb"))
    return pipe

def predict(model,text):
    label_dict = {0:'valid',1:'spam'}
    y_hat = model.predict(pd.DataFrame(text,columns=['text']))
    return pd.Series(y_hat).apply(lambda x : label_dict.get(x))

def save(model):
    filename = "../model/classifier.pickle"
    pickle.dump(model, open(filename, "wb"))

def train_save(X,y):
    model = train(X,y)
    save(model)
