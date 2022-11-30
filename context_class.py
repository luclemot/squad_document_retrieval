import json
import pandas as pd
from typing import List

"""This file creates a study class which tokenizes our data into classes :
a study has a list of texts,
a text has a title, multiple paragraphs,
a paragraph has a context, Q&As and a list of questions."""

with open("../squad1/train-v1.1.json", "r") as read_file:
    data = json.load(read_file)


class corpus:
    def __init__(self, list_texts):
        self.list_texts: List[text] = []

        self.load_data()

    def load_data(self):
        for i, temp in enumerate(data["data"]):
            self.list_texts.append(text(temp))

    def create_dataframe(self):
        """This function creates a pandas dataframe for our corpus, representing
        each qas with the according context, text title."""
        df = pd.json_normalize(
            data["data"],
            record_path = ["paragraphs", "qas"],
            meta = ['title', ["paragraphs",'context']],
            errors = "ignore"
        )
        df.rename(columns = {"paragraphs.context" : "context"}, inplace=True)
        return df


class text:
    def __init__(self, source):
        self.title = source["title"]
        self.paragraphs = source["paragraphs"]
        self.list_paragraphs: List[paragraph] = []

        self.load_paragraphs()

    def load_paragraphs(self):
        for i, para in enumerate(self.paragraphs):
            self.list_paragraphs.append(paragraph(para))


class paragraph:
    def __init__(self, source):
        self.context = source["context"]
        self.qas = source["qas"]
        self.questions = []

        self.load_questions()

    def load_questions(self):
        for i, q in enumerate(self.qas):
            self.questions.append(q["question"])


if __name__ == "__main__":
    test = corpus(data)
    print(test.list_texts[0].list_paragraphs[0].questions)
    print(test.create_dataframe().columns)
