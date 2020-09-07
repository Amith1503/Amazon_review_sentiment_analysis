''' 
IMPORTING PYTHONS RANDOM MODULE

'''
import random
import sys
import os
''' 
READING TXT FILES

'''

def read_data(data_path):
# Reading the negative reviews

	with open(os.path.join(data_path, 'neg.txt')) as f_neg:
		neg_lines = f_neg.readlines()

	# Reading the Positive reviews 
	with open(os.path.join(data_path, 'pos.txt')) as f_pos:
		pos_lines = f_pos.readlines()

	return neg_lines,pos_lines


''' PREPROCESSING FUNCTIONS '''

 # To remove special characters

def remove_spl_char(lines):
    spl_chars=['!','"','#','$','%','&','(',')','*','+','/',':',';','<','=','>','@','[','\\',']','^','~','{','|','`','}','\t','\n']
    for spl_char in spl_chars:
        lines = lines.replace(spl_char, '')
    return lines

# To Tokenize the sentences with stopwords

def pre_processor_with_sw(lines):
    lines= remove_spl_char(lines)
    words=lines.split()
    return words

# STOPWORDS LIST


stop_words=['i',
 'me',
 'my',
 'myself',
 'we',
 'our',
 'ours',
 'ourselves',
 'you',
 "you're",
 "you've",
 "you'll",
 "you'd",
 'your',
 'yours',
 'yourself',
 'yourselves',
 'he',
 'him',
 'his',
 'himself',
 'she',
 "she's",
 'her',
 'hers',
 'herself',
 'it',
 "it's",
 'its',
 'itself',
 'they',
 'them',
 'their',
 'theirs',
 'themselves',
 'what',
 'which',
 'who',
 'whom',
 'this',
 'that',
 "that'll",
 'these',
 'those',
 'am',
 'is',
 'are',
 'was',
 'were',
 'be',
 'been',
 'being',
 'have',
 'has',
 'had',
 'having',
 'do',
 'does',
 'did',
 'doing',
 'a',
 'an',
 'the',
 'and',
 'but',
 'if',
 'or',
 'because',
 'as',
 'until',
 'while',
 'of',
 'at',
 'by',
 'for',
 'with',
 'about',
 'against',
 'between',
 'into',
 'through',
 'during',
 'before',
 'after',
 'above',
 'below',
 'to',
 'from',
 'up',
 'down',
 'in',
 'out',
 'on',
 'off',
 'over',
 'under',
 'again',
 'further',
 'then',
 'once',
 'here',
 'there',
 'when',
 'where',
 'why',
 'how',
 'all',
 'any',
 'both',
 'each',
 'few',
 'more',
 'most',
 'other',
 'some',
 'such',
 'no',
 'nor',
 'not',
 'only',
 'own',
 'same',
 'so',
 'than',
 'too',
 'very',
 's',
 't',
 'can',
 'will',
 'just',
 'don',
 "don't",
 'should',
 "should've",
 'now',
 'd',
 'll',
 'm',
 'o',
 're',
 've',
 'y',
 'ain',
 'aren',
 "aren't",
 'couldn',
 "couldn't",
 'didn',
 "didn't",
 'doesn',
 "doesn't",
 'hadn',
 "hadn't",
 'hasn',
 "hasn't",
 'haven',
 "haven't",
 'isn',
 "isn't",
 'ma',
 'mightn',
 "mightn't",
 'mustn',
 "mustn't",
 'needn',
 "needn't",
 'shan',
 "shan't",
 'shouldn',
 "shouldn't",
 'wasn',
 "wasn't",
 'weren',
 "weren't",
 'won',
 "won't",
 'wouldn',
 "wouldn't"]

# To Tokenize the sentences without stopwords

def stop_words_removal(words_list):
    
    words_af_sw=[]
    for word in words_list:
        if word not in stop_words:
            words_af_sw.append(word)
    return words_af_sw

''' FUNCTIONS TO WRITE IN FILE '''

def file_writter_with_sw(filename,data):
    with open (filename,'w+') as f:
        for (tokens,labels) in data:
            f.write(",".join(tokens))
            f.write("\n")
        
def file_writter_without_sw(filename,data):
    with open (filename,'w+') as f:
        for (tokens,labels) in data:
            f.write(",".join(stop_words_removal(tokens)))
            f.write("\n")


def main(data_path):
	neg_lines,pos_lines=read_data(data_path)

	''' LOWERING AND TOKENIZING THE SENTENCE INTO WORDS '''

	tokenized_words=[]
	for neg_line in neg_lines:
	    neg_line=neg_line.lower()
	    tokenized_words.append([pre_processor_with_sw(neg_line),"neg"])
	    
	for pos_line in pos_lines:
	    pos_line=pos_line.lower()
	    tokenized_words.append([pre_processor_with_sw(pos_line),"pos"])

	''' SHUFFLING THE DATASET '''

	random.shuffle(tokenized_words)

	''' WRITING THE DATA WITH STOPWORDS AND LABELS TO FILE '''
	with open("processed_data/out_with_sw.csv",'w+') as f, open ("processed_data/label.csv",'w+') as g :
	    
	    for i in range (len(tokenized_words)):
	        f.write(",".join(tokenized_words[i][0]))
	        f.write("\n")
	        g.write((tokenized_words[i][1]))
	        g.write("\n")


	''' WRITING THE DATA WITHOUT STOPWORDS TO FILE
	'''
	out_no_sw=file_writter_without_sw("processed_data/out_without_sw.csv",tokenized_words)

	''' TRAIN/TEST/VALIDATION SPLIT OF DATA '''

	no_of_train_rows=int(0.8* len(tokenized_words))
	no_of_val_rows = int(0.1* len(tokenized_words))

	# SPLITTING THE DATA
	train_data = tokenized_words[:no_of_train_rows]
	val_data   = tokenized_words[no_of_train_rows : (no_of_train_rows + no_of_val_rows)]
	test_data  = tokenized_words[(no_of_train_rows + no_of_val_rows):]

	'''WRITING TO TXT FILE THE TRAIN/VAL/TEST DATA WITH STOPWORDS
	'''
	train_with_sw=file_writter_with_sw("processed_data/Train_data_with_sw.csv",train_data)
	val_with_sw=file_writter_with_sw("processed_data/Val_data_with_sw.csv",val_data)
	test_with_sw=file_writter_with_sw("processed_data/Test_data_with_sw.csv",test_data)

	''' WRITING TO TXT FILE THE TRAIN/VAL/TEST DATA WITHOUT STOPWORDS
	'''

	train_without_sw=file_writter_without_sw("processed_data/Train_data_without_sw.csv",train_data)
	val_without_sw=file_writter_without_sw("processed_data/Val_data_without_sw.csv",val_data)
	test_without_sw=file_writter_without_sw("processed_data/Test_data_without_sw.csv",test_data)


if __name__ == '__main__':
	main(sys.argv[1])

