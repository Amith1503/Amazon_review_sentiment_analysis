# Step 2:-

Write a python script using sklearn library to perform the following:
1. Train Multinomial Naïve Bayes (MNB) classifier to classify the documents in the Amazon corpus into positive and negative classes. Conduct experiments with the following conditions and report classification accuracy
you must use your training/validation/test data splits from Step 1. Train your models on the training set. You may only tune your models onyour validation set. Once the development is complete, run your classifier on your test
set.


# ACCURACIES:-

| StopWords Removed | Text Features    | Accuracy(Test Set) |
| ------------------|:----------------:| ------------------:|
| YES               | UNIGRAMS         | 80.68%             |
| YES               | BIGRAMS          | 78.97%             |
| YES               | UNIGRAMS +BIGRAMS| 82.47%             |
| NO                | UNIGRAMS         | 80.98%             |
| NO                | BIGRAMS          | 82.04%             |
| NO                | UNIGRAMS +BIGRAMS| 83.07%             |

# BRIEF REPORT:-

a) Which condition performed better: with or without stop Words?

Solution:-  When sentiment analysis is considered, with stopwords usually performs better. The reason behind is that few words in the stopwords removal list does play an important role when it comes to classification of reviews into positive and negative. For eg, if the review stated "not good" which is a negative review and since "not" is being added in the stopword removal list, the "not" is removed thus changing the sentence to positive, thereby decreasing the accuracy. Thus for sentiment analysis with stop words is best suited.

b) Which condition performed better: unigrams,bigrams or unigrams+bigrams?

Solution:- UNIGRAMS+BIGRAMS works better. As the length of n increases in n-gram, the amount of time the same n-grams seen in the test document reduces hence higher the value of n more it tends to overfit the training data. And if the value of n is very small, the model will fail to get the general contextual information, however, lower order n gram (unigram) works well for the data if the vocabulary is more and frequency of word occurances is less. Hence by combining UNIGRAMS+BIGRAMS leads to more features which will take care of the words for which the frequency is less and also helps the model to retain the contextual information thereby improving the accuracy.

# Execution Details:-

I have only placed the .py files in GitHub. The MSCI_641_Assignement_2.py file contains all the execution and the main.py is the file to be executed which has the driver of MSCI_641_Assignement_2.py.






