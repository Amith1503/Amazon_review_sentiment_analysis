# Step 3:-

1. Write a python script using genism library to train a Word2Vec model on the Amazon corpus.
2. Use genism library to get the most similar words to a given word. Find 20 most similar words to “good” and “bad”. Are the words most similar to “good” positive, and words most similar to “bad” negative? Why this is or isn’t the case? Explain your intuition
briefly (in 5-6 sentences).

# BRIEF REPORT ON MOST SIMILAR:-

## Are the words most similar to “good” positive, and words most similar to “bad” negative?

Yes, most of the words are positive with respect to "good" and negative with respect to word "bad". Word2Vec works by generating a vector for each word and the word vectors which have common contexts are almost close to each other. The "most similar" function works by using the cosine similarity between vectors and vectors which are relatively close are fetched and returned. It should also be noted that this is the reason why certain negative words like "bad" ,"terrible" matched for the word "good" and certain positive words like "good" matched for the word "bad".

# Exceution Details

I have only placed the .py files in GitHub. The MSCI_641_Assignment_3.py file contains all the execution and the main.py is the file to be executed which has the driver of MSCI_641_Assignment_3.py.

`inference.py` script generates the top-20 most similar words for a given word. It accepts one command-line argument
described below.
i. arg1: Path to a .txt file, which contains some words compiled for evaluation. There will be one word per line.



