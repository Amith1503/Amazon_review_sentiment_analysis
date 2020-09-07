# BRIEF REPORT ON MOST SIMILAR:-

## Are the words most similar to “good” positive, and words most similar to “bad” negative?

Yes, most of the words are positive with respect to "good" and negative with respect to word "bad". Word2Vec works by generating a vector for each word and the word vectors which have common contexts are almost close to each other. The "most similar" function works by using the cosine similarity between vectors and vectors which are relatively close are fetched and returned. It should also be noted that this is the reason why certain negative words like "bad" ,"terrible" matched for the word "good" and certain positive words like "good" matched for the word "bad".

# EXECUTION DETAILS:-

I have only placed the .py files in GitHub. The MSCI_641_Assignment_3.py file contains all the execution and the main.py is the file to be executed which has the driver of MSCI_641_Assignment_3.py. Kindly execute main.py. The main.py expects one argument to be passed from command line which is the path to the folder of assignment 1 which contains my tokenized out_with_sw.csv, eg:-"a1/data" please don't mention the file name, I have hardcoded the file name I created (out_with_sw.csv) in code using path.join. 
inference.py accepts a txt path which contains the words for which the most similar words to be found. 

# STUDENT DETAILS:-

Name :- Amith Nandakumar Student id:- 20859891 email:- a4nandak@uwaterloo.ca
