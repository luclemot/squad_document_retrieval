import pandas as pd
import matplotlib.pyplot as plt

from context_class import corpus, data

"""This file focuses on exploring our corpus :
do some paragraphs have a disproportionate number of questions?"""


def corpus_vis(temp):

    # Use the create_dataframe method on a corpus object.
    df = temp.create_dataframe()

    # Plot the question count distribution over all texts.
    plt.plot()
    df["title"].value_counts().plot(kind="barh")
    plt.yticks([])
    plt.show()

    # Plot the context count answering each question over all texts.
    plt.plot()
    df.groupby(["question"]).count()["context"].value_counts().plot(kind="bar")
    plt.show()

    print(max(df["question"].value_counts()))


if __name__ == "__main__":
    temp = corpus(data)
    corpus_vis(temp)
