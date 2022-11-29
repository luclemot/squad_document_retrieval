import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from random import randint

""" This file tries a first approach to solve our problem
by using a Tfid Vectorizer and the Nearest Neighbors learning method."""
from context_class import corpus, paragraph, data

# Defining the tf idf configurations
tfidf_configs = {
    "lowercase": True,
    "analyzer": "word",
    "stop_words": "english",
    "binary": True,
    "max_df": 0.9,
    "max_features": 10_000,
}

# Defining the number of documents to retrieve
retriever_configs = {"n_neighbors": 3, "metric": "cosine"}


def info_retrieval_nn(docs, text_title, text_content, question):
    # Defining our pipeline
    embedding = TfidfVectorizer(**tfidf_configs)
    retriever = NearestNeighbors(**retriever_configs)

    # Fitting our data on the target and identifier values.
    X = embedding.fit_transform(docs[text_content])
    retriever.fit(X, docs[text_title])

    # predict the most similar document
    X = embedding.transform([question])
    pred = []
    titles = []
    for i in range(3):
        title = retriever.kneighbors(X, return_distance=False)[0][i]
        titles.append(title)
        pred.append(docs.iloc[title]["context"])
    return titles, pred


def perf_metrics(data, pred_titles, question):
    # posi = lensemble des textes qui ont la question dans leur donnée
    posi = pd.DataFrame()
    """for i in range(len(data['qas'])):
        print(data.loc[i,"qas"])
        posi.append(data.loc[
        data.loc[i,"qas"]["question"] == question])"""
    # ce qu'on veut faire : retrouver le bon paragraphe et pas premier paragrpahe du bon texte.
    # ensuite, recréer objet paragraphe, call ses questions. tadam.
    # for title in pred_titles :
    # text =
    # context =
    # qas =
    # para = paragraph({'context':context, 'qas', qas})
    # tp = len(intersection 'test' et pred)
    # fp = ceux dans pred et pas dans 'test'
    # on va étudier la précision, pour estimer le nombre de vrai relevants dans les réponses.
    tp = posi.shape[0]
    fp = 1
    prec = tp / (tp + fp)
    return prec


if __name__ == "__main__":
    test = corpus(data)
    df = test.create_dataframe()

    documents = df[["context", "title"]].drop_duplicates().reset_index(drop=True)

    n = randint(0, len(test.list_texts))
    m = randint(0, len(test.list_texts[n].list_paragraphs))
    p = randint(0, len(test.list_texts[n].list_paragraphs[m].questions))
    text = test.list_texts[n].list_paragraphs[m].questions[p]
    print(text)

    titles, pred = info_retrieval_nn(documents, "title", "context", text)
    print(pred[0])
    print(perf_metrics(df, pred, text))
