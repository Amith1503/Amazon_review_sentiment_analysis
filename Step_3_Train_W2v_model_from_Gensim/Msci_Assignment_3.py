# Importing Libraries
import os
import sys
import nltk
from gensim.models import Word2Vec

#Read Data
def read_path(file_path):
    with open (file_path) as f:
        lines=f.readlines()
        
    return [' '.join(line.strip().split(",")) for line in lines]

def read_data(file_path):
    out_data_with_sw= read_path(os.path.join(file_path,"out_with_sw.csv"))
    
    return out_data_with_sw


def word_nltk(sentences):
    return [nltk.word_tokenize(sentence) for sentence in sentences]


def W2V(sentence):
    model= Word2Vec(sentence, size=100, window=5, min_count=2, workers=4)
    return model


def word_similarity(model,word):
    return model.wv.most_similar(word,topn=20) 


def main(data_path):
    sentences= read_data(data_path)
    word_sentences= word_nltk(sentences)

    # RUNNING THE MODEL AND SAVING THE MODEL
    W2V_model= W2V(word_sentences)
    W2V_model.save("a3/data/w2v.model")

    # PRINTING TOP 20 WORDS RELATED TO GOOD (NOTE:- PRINTING ONLY THE WORDS)
    similar_words_good= word_similarity(W2V_model,"good")
    print("The top 20 words similar to GOOD : ") 
    [print(similar_word) for (similar_word,prob) in similar_words_good]

    # PRINTING TOP 20 WORDS RELATED TO GOOD (NOTE:- PRINTING ONLY THE WORDS)
    similar_words_bad= word_similarity(W2V_model,"bad")
    print("The top 20 words similar to BAD : ")
    [print(similar_word) for (similar_word,prob) in similar_words_bad]

if __name__ == '__main__':
    main(sys.argv[1])
