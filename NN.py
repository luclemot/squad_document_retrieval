import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from random import choice, choices

""" This file tries a first approach to solve our problem
by using a Tfid Vectorizer and the Nearest Neighbors learning method."""

from context_class import corpus, data

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


def doc_retrieval_nn(docs, question):
    # Defining our pipeline
    embedding = TfidfVectorizer(**tfidf_configs)
    retriever = NearestNeighbors(**retriever_configs)

    # Fitting our data on the target and identifier values.
    X = embedding.fit_transform(docs["context"])
    retriever.fit(X, docs["title"])

    # predict the most similar document
    X = embedding.transform([question])
    index = retriever.kneighbors(X, return_distance=False)[0][0]
    pred = docs.iloc[index]["context"]
    title = docs.iloc[index]["title"]
    return title, pred


def perf(data, question, pred_context=None, pred_title=None):
    # Defining if the predicted context or predicted text title answers the given question.
    if pred_context:
        temp = data.loc[data["context"] == pred_context]
    if pred_title:
        temp = data.loc[data["title"] == pred_title]
    posi = temp.loc[temp["question"] == question]
    tp = posi.shape[0]
    return tp > 0


def precision(data, t=False, c=False):
    # Computing the precision, equal to true positives over all positives,
    # to see how many useless documents we're returning.
    T, F = 0, 0
    documents = data[["context", "title"]].drop_duplicates().reset_index(drop=True)
    # Create a unique list of questions, and pick 100 at random.
    questions = list(data["question"].unique())
    questions = choices(questions, k=100)
    for q in questions:
        # find the best prediction for this question, and add to T or F if it's a match.
        title, pred = doc_retrieval_nn(documents, q)
        if t:
            if perf(data, q, pred_title=title):
                T += 1
            else:
                F += 1
        elif c:
            if perf(data, q, pred_context=pred):
                T += 1
            else:
                F += 1
    pred = T / (T + F)
    return pred


if __name__ == "__main__":

    # Transform our data into a corpus object.
    corp = corpus(data)
    # Apply the create_dataframe method.
    df = corp.create_dataframe()

    documents = df[["context", "title"]].drop_duplicates().reset_index(drop=True)

    # Select a random question.
    trial_text = choice(corp.list_texts)
    trial_paragraph = choice(trial_text.list_paragraphs)
    trial_question = choice(trial_paragraph.questions)
    print(trial_question)

    # Retrieve the 'closest' document based on our NN method.
    title, pred = doc_retrieval_nn(documents, trial_question)
    print(title)
    print(pred)
    # Print a boolean to indicate if it's a match.
    print(perf(df, trial_question, pred_context=pred))

    # print(precision(df, c=True))
    # On a run of 100 questions, the precision based on context is 0.47.
    # print(precision(df, t=True))
    # On a run of 100 questions, the precision based on title is 0.5.
