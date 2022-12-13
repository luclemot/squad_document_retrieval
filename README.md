# SQUAD - Document Retrieval test

## Objective of this repo :
This project was my first interaction with Natural Language Processing. I was asked to create a code that, given a question and a corpus, would return the text that is most likely to answer the question. The link to the original database is [here](https://rajpurkar.github.io/SQuAD-explorer/).

## File structure :
- The [context_class.py](context_class.py) file creates our corpus object and parses texts with their according contexts, questions, and answers.
- The [explo.py](explo.py) file explores and visualizes the data distribution in the given corpus.
- The [NN.py](NN.py) file offers a first solution to our problem. Using a NN classification and a tfidf embedding, we are able to return the closest text to each given question.
- The [validation.py](validation.py) file gives the user the opportunity to test our classification on a random or a selected question, including questions in the validation set that don't have a target value.

## Performance and next steps :
The current results are the following : 
- An accuracy of `0.53` based on the returned context
- An accuracy of `0.63` based on the returned text title
If I had more time to work on this, here are the first couple of things I would try :
- I would change it so that my classification doesn't only return one context but 3 (or the optimal number). The goal is to ensure I improve my accuracy, or change the performance metric altogether into a k-accuracy. It is true that, alike on a search engine, the user wants the right document to pop up, but doesn't care if it's not the first.
- I would try to switch embeddings, and perhaps go the route of word2vec.
- It would be interesting to use a Bert model or another well established NLP tool.