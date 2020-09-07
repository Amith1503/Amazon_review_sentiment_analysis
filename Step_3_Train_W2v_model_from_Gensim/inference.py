import sys
from Msci_Assignment_3  import word_similarity
from gensim.models import Word2Vec

def path(data):
    
    with open(data) as f:
        ev= f.readlines()
        
    return[ line.strip("\n") for line in ev]

def main(data_path):
	w2v= Word2Vec.load("a3/data/w2v.model")
	data_read= path(data_path)


	# PRINTING ONLY THE TOP 20 WORDS FROM THE FILE CONTAINING THE WORDS IN EACH LINE 
	for i in range (len(data_read)):
		related_words = word_similarity(w2v,data_read[i])
		print("The top 20 Revelant word for ",data_read[i])
		[print(similar_word) for (similar_word,prob) in related_words]


if __name__ == '__main__':
	main(sys.argv[1])