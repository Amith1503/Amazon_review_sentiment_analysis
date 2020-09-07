# Importing Libraries
import os
import sys
import numpy as np
from gensim.models import Word2Vec

from keras.layers import *
from keras.models import Sequential, load_model, save_model
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.text import one_hot

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score




def main(data_path):

	train_data_file = os.path.join(data_path,"Train_data_with_sw.csv")
	test_data_file = os.path.join(data_path,"Test_data_with_sw.csv")
	val_data_file = os.path.join(data_path,"Val_data_with_sw.csv") 
	labels_file = os.path.join(data_path,"label.csv")
	wv_model_file = "a3/data/w2v.model"

	def read_csv(file):    
	    with open(file, "r") as f:
	        content = f.readlines()
	    content = [line.strip("\n") for line in content]
	    content = [line.split(",") for line in content]
	    return content

    # read content of the data files
	train_data = read_csv(train_data_file)
	test_data = read_csv(test_data_file)
	val_data = read_csv(val_data_file)
	labels = read_csv(labels_file)

	labels_dict = {'pos': 0, 'neg': 1}
	print(labels_dict)

	# split labels
	train_labels = labels[:640000]
	val_labels = labels[640000:(640000 + 80000)]
	test_labels = labels[(640000 + 80000):(640000 + 80000 + 80000)]

	# calculate maximum sentence length
	MAX_LENGTH = np.max([len(td) for td in train_data] + [len(td) for td in test_data] + [len(td) for td in val_data])

	# Load trained word2vec model
	w2v = Word2Vec.load(wv_model_file)
	EMBEDDING_DIM = w2v.wv.vector_size

	# create vocabulary
	vocabulary = list()
	for sentence in train_data + val_data + test_data:
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




	X_train = np.array([sentence_idx(sentence) for sentence in train_data], dtype = np.int32)
	X_test = np.array([sentence_idx(sentence) for sentence in test_data], dtype = np.int32)
	X_val = np.array([sentence_idx(sentence) for sentence in val_data], dtype = np.int32)

	y_train = np.array([convert_labels(l) for l in train_labels], dtype = np.int8)
	y_test = np.array([convert_labels(l) for l in test_labels], dtype = np.int8)
	y_val = np.array([convert_labels(l) for l in val_labels], dtype = np.int8)

	# Build models

	# RELU MODEL
	def build_model_relu(word_embeddings):
	    model=Sequential()
	    model.add(Embedding(word_embeddings.shape[0],word_embeddings.shape[1],weights=[word_embeddings],trainable=False,input_length=MAX_LENGTH))
	    model.add(LSTM(128,activation="relu"))
	    model.add(Dense(1,activation='sigmoid'))
	    return model

	def build_model_relu_d02(word_embeddings):
	    model=Sequential()
	    model.add(Embedding(word_embeddings.shape[0],word_embeddings.shape[1],weights=[word_embeddings],trainable=False,input_length=MAX_LENGTH,embeddings_regularizer=L1L2(l1=0.0, l2=0.2)))
	    model.add(Dropout(0.2))
	    model.add(LSTM(128,activation="relu",bias_regularizer=L1L2(l1=0.0, l2=0.2)))
	    model.add(Dropout(0.2))
	    model.add(Dense(1,activation='sigmoid'))
	    return model

	def build_model_relu_l2(word_embeddings):
	    model=Sequential()
	    model.add(Embedding(word_embeddings.shape[0],word_embeddings.shape[1],weights=[word_embeddings],trainable=False,input_length=MAX_LENGTH,embeddings_regularizer=L1L2(l1=0.0, l2=0.2)))
	    # model.add(Dropout(0.2))
	    model.add(LSTM(128,activation="relu",bias_regularizer=L1L2(l1=0.0, l2=0.2)))
	    # model.add(Dropout(0.2))
	    model.add(Dense(1,activation='sigmoid'))
	    return model

	def build_model_relu_d(word_embeddings):
	    model=Sequential()
	    model.add(Embedding(word_embeddings.shape[0],word_embeddings.shape[1],weights=[word_embeddings],trainable=False,input_length=MAX_LENGTH))
	    model.add(Dropout(0.2))
	    model.add(LSTM(128,activation="relu"))
	    model.add(Dropout(0.2))
	    model.add(Dense(1,activation='sigmoid'))
	    return model

	def build_model_relu_d04(word_embeddings):
	    model=Sequential()
	    model.add(Embedding(word_embeddings.shape[0],word_embeddings.shape[1],weights=[word_embeddings],trainable=False,input_length=MAX_LENGTH,embeddings_regularizer=L1L2(l1=0.0, l2=0.4)))
	    model.add(Dropout(0.4))
	    model.add(LSTM(128,activation="relu",bias_regularizer=L1L2(l1=0.0, l2=0.4)))
	    model.add(Dropout(0.4))
	    model.add(Dense(1,activation='sigmoid'))
	    return model

	def build_model_tanh(word_embeddings):
	    model=Sequential()
	    model.add(Embedding(word_embeddings.shape[0],word_embeddings.shape[1],weights=[word_embeddings],trainable=False,input_length=MAX_LENGTH))
	    model.add(LSTM(128,activation="tanh"))
	    model.add(Dense(1,activation='sigmoid'))
	    return model

	def build_model_tanh_d02(word_embeddings):
	    model=Sequential()
	    model.add(Embedding(word_embeddings.shape[0],word_embeddings.shape[1],weights=[word_embeddings],trainable=False,input_length=MAX_LENGTH,embeddings_regularizer=L1L2(l1=0.0, l2=0.2)))
	    model.add(Dropout(0.2))
	    model.add(LSTM(128,activation="tanh",bias_regularizer=L1L2(l1=0.0, l2=0.2)))
	    model.add(Dropout(0.2))
	    model.add(Dense(1,activation='sigmoid'))
	    return model

	def build_model_tanh_l2(word_embeddings):
	    model=Sequential()
	    model.add(Embedding(word_embeddings.shape[0],word_embeddings.shape[1],weights=[word_embeddings],trainable=False,input_length=MAX_LENGTH,embeddings_regularizer=L1L2(l1=0.0, l2=0.2)))
	    # model.add(Dropout(0.2))
	    model.add(LSTM(128,activation="tanh",bias_regularizer=L1L2(l1=0.0, l2=0.2)))
	    # model.add(Dropout(0.2))
	    model.add(Dense(1,activation='sigmoid'))
	    return model

	def build_model_tanh_d(word_embeddings):
	    model=Sequential()
	    model.add(Embedding(word_embeddings.shape[0],word_embeddings.shape[1],weights=[word_embeddings],trainable=False,input_length=MAX_LENGTH))
	    model.add(Dropout(0.2))
	    model.add(LSTM(128,activation="tanh"))
	    model.add(Dropout(0.2))
	    model.add(Dense(1,activation='sigmoid'))
	    return model

	def build_model_tanh_d04(word_embeddings):
	    model=Sequential()
	    model.add(Embedding(word_embeddings.shape[0],word_embeddings.shape[1],weights=[word_embeddings],trainable=False,input_length=MAX_LENGTH,embeddings_regularizer=L1L2(l1=0.0, l2=0.4)))
	    model.add(Dropout(0.4))
	    model.add(LSTM(128,activation="tanh",bias_regularizer=L1L2(l1=0.0, l2=0.4)))
	    model.add(Dropout(0.4))
	    model.add(Dense(1,activation='sigmoid'))
	    return model

	def build_model_sigmoid(word_embeddings):
	    model=Sequential()
	    model.add(Embedding(word_embeddings.shape[0],word_embeddings.shape[1],weights=[word_embeddings],trainable=False,input_length=MAX_LENGTH))
	    model.add(LSTM(128,activation="sigmoid"))
	    model.add(Dense(1,activation='sigmoid'))
	    return model

	def build_model_sigmoid_d02(word_embeddings):
	    model=Sequential()
	    model.add(Embedding(word_embeddings.shape[0],word_embeddings.shape[1],weights=[word_embeddings],trainable=False,input_length=MAX_LENGTH,embeddings_regularizer=L1L2(l1=0.0, l2=0.2)))
	    model.add(Dropout(0.2))
	    model.add(LSTM(128,activation="sigmoid",bias_regularizer=L1L2(l1=0.0, l2=0.2)))
	    model.add(Dropout(0.2))
	    model.add(Dense(1,activation='sigmoid'))
	    return model

	def build_model_sigmoid_l2(word_embeddings):
	    model=Sequential()
	    model.add(Embedding(word_embeddings.shape[0],word_embeddings.shape[1],weights=[word_embeddings],trainable=False,input_length=MAX_LENGTH,embeddings_regularizer=L1L2(l1=0.0, l2=0.2)))
	    # model.add(Dropout(0.2))
	    model.add(LSTM(128,activation="sigmoid",bias_regularizer=L1L2(l1=0.0, l2=0.2)))
	    # model.add(Dropout(0.2))
	    model.add(Dense(1,activation='sigmoid'))
	    return model

	def build_model_sigmoid_d(word_embeddings):
	    model=Sequential()
	    model.add(Embedding(word_embeddings.shape[0],word_embeddings.shape[1],weights=[word_embeddings],trainable=False,input_length=MAX_LENGTH))
	    model.add(Dropout(0.2))
	    model.add(LSTM(128,activation="sigmoid"))
	    model.add(Dropout(0.2))
	    model.add(Dense(1,activation='sigmoid'))
	    return model

	def build_model_sigmoid_d04(word_embeddings):
	    model=Sequential()
	    model.add(Embedding(word_embeddings.shape[0],word_embeddings.shape[1],weights=[word_embeddings],trainable=False,input_length=MAX_LENGTH,embeddings_regularizer=L1L2(l1=0.0, l2=0.4)))
	    model.add(Dropout(0.4))
	    model.add(LSTM(128,activation="sigmoid",bias_regularizer=L1L2(l1=0.0, l2=0.4)))
	    model.add(Dropout(0.4))
	    model.add(Dense(1,activation='sigmoid'))
	    return model
	def train_model(model, model_name, epochs = 5, batch_size = 256):
	    callback = ModelCheckpoint(model_name, monitor='val_accuracy', save_best_only=True)
	    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
	    model.fit(X_train, y_train, epochs = epochs, batch_size = batch_size, validation_data = (X_val, y_val), callbacks = [callback])
	    print(model.evaluate(X_test, y_test))

	# start training
	model = build_model_relu(word_embeddings)
	train_model(model, os.path.join(data_path,"Relu_v0.hdf5"))
	del model

	model_1 = build_model_relu_d02(word_embeddings)
	train_model(model_1, os.path.join(data_path,"Relu_d02.hdf5"))
	del model_1

	model_2 = build_model_relu_d04(word_embeddings)
	train_model(model_2, os.path.join(data_path,"Relu_d04.hdf5"))
	del model_2

	model_3 = build_model_tanh(word_embeddings)
	train_model(model_3, os.path.join(data_path,"Tanh_v0.hdf5"))
	del model_3

	model_4 = build_model_tanh_d02(word_embeddings)
	train_model(model_4, os.path.join(data_path,"Tanh_d02.hdf5"))
	del model_4

	model_5 = build_model_tanh_d04(word_embeddings)
	train_model(model_5, os.path.join(data_path,"Tanh_d04.hdf5"))
	del model_5

	model_6 = build_model_sigmoid(word_embeddings)
	train_model(model_6, os.path.join(data_path,"Sigmoid_v0.hdf5"))
	del model_6

	model_7 = build_model_sigmoid_d02(word_embeddings)
	train_model(model_7, os.path.join(data_path,"Sigmoid_d02.hdf5"))
	del model_7

	model_8 = build_model_sigmoid_d04(word_embeddings)
	train_model(model_8, os.path.join(data_path,"Sigmoid_d04.hdf5"))
	del model_8

	model_9 = build_model_relu_l2(word_embeddings)
	train_model(model_9, os.path.join(data_path,"relu_l2.hdf5"))
	del model_9

	model_10 = build_model_relu_d(word_embeddings)
	train_model(model_10, os.path.join(data_path,"relu_d.hdf5"))
	del model_10

	model_11 = build_model_sigmoid_l2(word_embeddings)
	train_model(model_11, os.path.join(data_path,"Sigmoid_l2.hdf5"))
	del model_11

	model_12 = build_model_sigmoid_d(word_embeddings)
	train_model(model_12, os.path.join(data_path,"relu_l2.hdf5"))
	del model_12

	model_13 = build_model_tanh_l2(word_embeddings)
	train_model(model_13, os.path.join(data_path,"tanh_l2.hdf5"))
	del model_13

	model_14 = build_model_tanh_d(word_embeddings)
	train_model(model_14, os.path.join(data_path,"tanh_d.hdf5"))
	del model_14





if __name__ == '__main__':
	main(sys.argv[1])

