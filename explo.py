import pandas as pd
import matplotlib.pyplot as plt

from context_class import corpus, data

"""This file focuses on exploring our corpus :
do some paragraphs have a disproportionate number of questions?
keywords?"""


def corpus_vis(temp):
    qs = []
    titles = []
    for text in temp.list_texts:
        for para in text.list_paragraphs:
            for i, q in enumerate(para.questions):
                qs.append(q)
                titles.append(text.title)
    Qs = pd.DataFrame(titles, qs)
    Qs.reset_index(inplace=True)
    Qs.rename(columns={"index": "question", 0: "text_title"}, inplace=True)

    plt.plot()
    Qs["text_title"].value_counts().plot(kind="barh")
    plt.yticks([])
    plt.show()


if __name__ == "__main__":
    temp = corpus(data)
    corpus_vis(temp)
