import sys
# from MSCI_Assignment_4 import read_path,one_hot_repr,y_labels,testing,read_lables
from gensim.models import Word2Vec
import numpy as np
import tensorflow as tf
from keras.models import Sequential, load_model, save_model
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

def read_data(data_path):
# Reading the negative reviews

	with open(data_path) as f:
		lines = f.readlines()

	return lines

def remove_spl_char(lines):
    spl_chars=['!','"','#','$','%','&','(',')','*','+','/',':',';','<','=','>','@','[','\\',']','^','~','{','|','`','}','\t','\n']
    for spl_char in spl_chars:
        lines = lines.replace(spl_char, '')
    return lines

def pre_processor_with_sw(lines):
    lines= remove_spl_char(lines)
    words=lines.split()
    return words

def test_model(model_path, X_test):
    model = load_model(model_path)
    # try:
    #     print("Test accuracy is", model.evaluate(X_test, y_test))
    # except:
    #     pass
    y_pred = model.predict_classes(X_test)
    return y_pred


def main(data_path,model_path):
	test_lines= read_data(data_path)

	test_data=[]

	MAX_LENGTH=43




	for test_line in test_lines:
	    test_line=test_line.lower()
	    test_data.append(pre_processor_with_sw(test_line))


	# Load trained word2vec model
	w2v = Word2Vec.load("a3/data/w2v.model")
	EMBEDDING_DIM = w2v.wv.vector_size

	# create vocabulary
	vocabulary = list()
	for sentence in test_data:
	    vocabulary += sentence
	vocabulary = sorted(list(set(vocabulary)))

	# create tokens for padding and unknown
	tokens = ['<PAD>', '<UNK>']

	# get usable word embeddings
	word_embeddings = list()
	word_embeddings.append(np.zeros(EMBEDDING_DIM))
	word_embeddings.append(np.ones(EMBEDDING_DIM))

	IDX_WORD = tokens

	for word in vocabulary:
	    try:
	        word_vector = w2v.wv.get_vector(word)
	        word_embeddings.append(word_vector)
	        IDX_WORD.append(word)
	    except:
	        continue

	WORD_IDX = dict([(word,idx) for idx,word in enumerate(IDX_WORD)])
	VOCAB_SIZE = len(IDX_WORD)
	word_embeddings = np.array(word_embeddings)
	del w2v

	def pad_sequence(sentence, pad_item = "<PAD>"):
	    while len(sentence) < MAX_LENGTH:
	        sentence += [pad_item]
	    return sentence

	def idx_onehot(word_idx):
	    onehot = [0]*VOCAB_SIZE
	    onehot[word_idx] = 1
	    return onehot

	def sentence_onehot(sentence):
	    onehot_vector = list()
	    sentence = pad_sequence(sentence)
	    for word in sentence:
	        try:
	            word_idx = WORD_IDX[word]
	        except:
	            word_idx = WORD_IDX['<UNK>']
	        onehot_vector.append(idx_onehot(word_idx))
	    return onehot_vector

	def sentence_idx(sentence):
	    sentence = pad_sequence(sentence)
	    sentence_indices = list()
	    for word in sentence:
	        try:
	            word_idx = WORD_IDX[word]
	        except:
	            word_idx = WORD_IDX['<UNK>']
	        sentence_indices.append(word_idx)
	    return sentence_indices
	        
	def convert_labels(labels_list):
	    return np.array([labels_dict[label] for label in labels_list], dtype = np.int8)


	X_test = np.array([sentence_idx(sentence) for sentence in test_data], dtype = np.int32)

	y_pred = test_model(model_path,X_test)

	# print(y_pred)
	for i in y_pred:
		if (i==0):
			print("positive")
		elif(i==1):
			print("negative")


if __name__ == '__main__':
	main(sys.argv[1],sys.argv[2])
