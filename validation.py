from NN import doc_retrieval_nn
from context_class import corpus
from random import choice
import json

if __name__ == "__main__":

    with open("../squad1/train-v1.1.json", "r") as read_file:
        valid = json.load(read_file)

    # Transform our data into a corpus object.
    corp = corpus(valid)
    # Apply the create_dataframe method.
    df = corp.create_dataframe()

    documents = df[["context", "title"]].drop_duplicates().reset_index(drop=True)

    message = input("Enter a question or type random : ")
    if message == "random" :
        # Select a random question.
        trial_text = choice(corp.list_texts)
        trial_paragraph = choice(trial_text.list_paragraphs)
        trial_question = choice(trial_paragraph.questions)
    else:
        trial_question = message
    print(trial_question)

    # Retrieve the 'closest' document based on our NN method.
    title, pred = doc_retrieval_nn(documents, trial_question)
    print(title)
    print(pred)