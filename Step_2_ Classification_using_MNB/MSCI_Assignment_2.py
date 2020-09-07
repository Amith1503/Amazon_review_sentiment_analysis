import sys
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB


def read_path(file_path):
    with open (file_path) as f:
        read_data= f.readlines()
        
        
    return [' '.join(line.strip().split(",")) for line in read_data]

def read_label(file_path):
    with open(file_path) as f:
        read_line= f.readlines()
        
    return read_line

def word_prob_tfidf(dataset,ngram,vocab_model):
    if vocab_model==None:
        if ngram=="unigram":
            Tf_idf= TfidfVectorizer(ngram_range = (1,1))
            X_data = Tf_idf.fit_transform(dataset)

        if ngram=="bigram":
            Tf_idf= TfidfVectorizer(ngram_range = (2,2))
            X_data = Tf_idf.fit_transform(dataset)

        if ngram=="both":
            Tf_idf= TfidfVectorizer(ngram_range = (1,2))
            X_data = Tf_idf.fit_transform(dataset)
            
    else:
        
        if ngram=="unigram":
            Tf_idf= TfidfVectorizer(ngram_range = (1,1),vocabulary=vocab_model.vocabulary_)
            X_data = Tf_idf.fit_transform(dataset)

        if ngram=="bigram":
            Tf_idf= TfidfVectorizer(ngram_range = (2,2),vocabulary=vocab_model.vocabulary_)
            X_data = Tf_idf.fit_transform(dataset)

        if ngram=="both":
            Tf_idf= TfidfVectorizer(ngram_range = (1,2),vocabulary=vocab_model.vocabulary_)
            X_data = Tf_idf.fit_transform(dataset)
        

    return Tf_idf,X_data

def read_data_sw(file_path):
    x_train = read_path(os.path.join(file_path,"Train_data_with_sw.csv"))
    x_val =read_path(os.path.join(file_path,"Val_data_with_sw.csv"))
    x_test=read_path(os.path.join(file_path,"Test_data_with_sw.csv"))
    label = read_label(os.path.join (file_path,"label.csv"))

    y_train = label[:len(x_train)]
    y_val =label[len(x_train):(len(x_train)+len(x_val))]
    y_test =label[(len(x_train)+len(x_val)):]
  
    
    return x_train,x_val,x_test,y_train,y_val,y_test


def read_data_without_sw(file_path):
    x_train = read_path(os.path.join(file_path,"Train_data_without_sw.csv"))
    x_val =read_path(os.path.join(file_path,"Val_data_without_sw.csv"))
    x_test=read_path(os.path.join(file_path,"Test_data_without_sw.csv"))
    label = read_label(os.path.join (file_path,"label.csv"))

    y_train = label[:len(x_train)]
    y_val =label[len(x_train):(len(x_train)+len(x_val))]
    y_test =label[(len(x_train)+len(x_val)):]

    
    return x_train,x_val,x_test,y_train,y_val,y_test

def train_model(x_data,y_data):
    
    senti_detect_model = MultinomialNB().fit(x_data, y_data)
    
    return senti_detect_model


def predict(model,test_data):
    y_pred= model.predict(test_data)
    return y_pred


def metrics(y_true,y_pred):
    report= classification_report(y_true,y_pred)
    accuracy= accuracy_score(y_true,y_pred)
    
    return report,accuracy

def unigram(x_train,x_val,x_test,y_train,y_val,y_test):
    tf_idf,X_data=word_prob_tfidf(x_train,"unigram",None)
    model=train_model(X_data,y_train)

    #predict val Please uncommnet the below 2 lines for unigram predict validation set
    # tf_idf_val,X_val=word_prob_tfidf(x_val,"unigram",tf_idf)
    # y_pred_val= predict(model,X_val)

    #predict test
    tf_idf_test,X_test=word_prob_tfidf(x_test,"unigram",tf_idf)
    y_pred_test= predict(model,X_test)



    #val metrics
    # report,accuracy=metrics(y_val,y_pred_val)
#     print("report",report)
#     print("accuracy",accuracy)

    #test metrics
    report,accuracy=metrics(y_test,y_pred_test)

    return report,accuracy


def bigram(x_train,x_val,x_test,y_train,y_val,y_test):
    tf_idf,X_data=word_prob_tfidf(x_train,"bigram",None)
    model=train_model(X_data,y_train)

    #predict val Please uncommnet the below 2 lines for bigram predict validation set
    # tf_idf_val,X_val=word_prob_tfidf(x_val,"bigram",tf_idf)
    # y_pred_val= predict(model,X_val)

    #predict test
    tf_idf_test,X_test=word_prob_tfidf(x_test,"bigram",tf_idf)
    y_pred_test= predict(model,X_test)



    #val metrics
    # report,accuracy=metrics(y_val,y_pred_val)
#     print("report",report)
#     print("accuracy",accuracy)

    #test metrics
    report,accuracy=metrics(y_test,y_pred_test)
    return report,accuracy


def both(x_train,x_val,x_test,y_train,y_val,y_test):
    tf_idf,X_data=word_prob_tfidf(x_train,"both",None)
    model=train_model(X_data,y_train)

    #predict val Please uncommnet the below 2 lines for unigram+bigram predict validation set
    tf_idf_val,X_val=word_prob_tfidf(x_val,"both",tf_idf)
    y_pred_val= predict(model,X_val)

    #predict val
    tf_idf_test,X_test=word_prob_tfidf(x_test,"both",tf_idf)
    y_pred_test= predict(model,X_test)



    #val metrics
    # report,accuracy=metrics(y_val,y_pred_val)
#     print("report",report)
#     print("accuracy",accuracy)

    #test metrics
    report,accuracy=metrics(y_test,y_pred_test)

    return report,accuracy


def main(data_path):

    # WITH STOP WORDS

    x_train,x_val,x_test,y_train,y_val,y_test = read_data_sw(data_path)

    unigram_report,unigram_accuracy= unigram(x_train,x_val,x_test,y_train,y_val,y_test)
    print("*** UNIGRAM WITH STOP WORDS***")
    print(unigram_report)
    print("unigram_accuracy_SW : ",unigram_accuracy)
    print()

    bigram_report,bigram_accuracy= bigram(x_train,x_val,x_test,y_train,y_val,y_test)
    print("*** BIGRAM WITH STOP WORDS***")
    print(bigram_report)
    print("bigram_accuracy_SW : ",bigram_accuracy)
    print()
    
    both_report,both_accuracy= both(x_train,x_val,x_test,y_train,y_val,y_test)

    print("*** UNIGRAM + BIGRAM WITH STOP WORDS***")
    print(both_report)
    print("UNIGRAM + BIGRAM_SW : ",both_accuracy)
    print()


    # WITHOUT STOP WORDS

    x_train,x_val,x_test,y_train,y_val,y_test = read_data_without_sw(data_path)

    unigram_report,unigram_accuracy= unigram(x_train,x_val,x_test,y_train,y_val,y_test)
    print("*** UNIGRAM WITHOUT STOP WORDS***")
    print(unigram_report)
    print("unigram_accuracy_without_SW : ",unigram_accuracy)
    print()

    bigram_report,bigram_accuracy= bigram(x_train,x_val,x_test,y_train,y_val,y_test)
    print("*** BIGRAM WITHOUT STOP WORDS***")
    print(bigram_report)
    print("bigram_accuracy_without_SW : ",bigram_accuracy)
    print()

    both_report,both_accuracy= both(x_train,x_val,x_test,y_train,y_val,y_test)
    print("*** UNIGRAM + BIGRAM WITHOUT STOP WORDS***")
    print(both_report)
    print("UNIGRAM + BIGRAM_without_SW : ",both_accuracy)


if __name__ == '__main__':
    main(sys.argv[1])

