from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd

from context_class import corpus, data


""" This file tries a first classification approach on a tokenized view of the contexts and 
questions. """


temp = corpus(data)


def create_context_set(temp):
    df = []
    for text in temp.list_texts:
        for para in text.list_paragraphs:
            df.append(para.context)
            # Ici, on peut preprocess le format du 'contexte' si besoin.
    return df


def create_train_clf(temp):
    qs = []
    titles = []
    for text in temp.list_texts:
        for para in text.list_paragraphs:
            for i, q in enumerate(para.questions):
                qs.append(q)
                titles.append(para.context)
    train_set = pd.DataFrame(titles, qs)
    train_set.reset_index(inplace=True)
    train_set.rename(columns={"index": "question", 0: "context"}, inplace=True)
    return train_set

train_set = create_train_clf(temp)

x_train, x_test = train_test_split(train_set, test_size = 0.2)


# reprendre classif de bechdel, puis essayer doc2vec
# on va classifier sur les features. classif : naive bayes


# create train/test split, 0.2. Besoin de stratifier? voir répartition des contextes
