import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import string
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
import time
from subprocess import Popen, PIPE
from Dataset import CollectData
import pickle
import re
import datetime

def exec_applescript(script):
	p = Popen(['osascript', '-e', script])
	
class ChatBot():

	def __init__(self): 
		
		self.model = tf.saved_model.load('model')
		with open('tokenizer.pickle', 'rb') as handle:
		    self.tokenizer = pickle.load(handle)
		    
		with open('max_sequence_len.pickle', 'rb') as handle:
		    self.max_sequence_len = pickle.load(handle)
		    
		    
		self.msg = """
		on is_running(appName)
			tell application "System Events" to (name of processes) contains appName
		end is_running

		set CR to ASCII character of 13
		set ENTR to 36
		set TB to 48

		set safRunning to is_running("World of Warcraft")

		if safRunning is true then
			try
				tell application "System Events"
					tell application "World of Warcraft" to activate
					key code ENTR
					keystroke "/2 " & "{TEXT}" & CR
				end tell
			end try
		end if

		"""
		
	def reload_model(self):
		self.model = tf.saved_model.load('model')
		with open('tokenizer.pickle', 'rb') as handle:
		    self.tokenizer = pickle.load(handle)
		    
		with open('max_sequence_len.pickle', 'rb') as handle:
		    self.max_sequence_len = pickle.load(handle)

	def execute(self):
		cd = CollectData()
		tmp = cd.collectchat()
		print('collecting data')
		time.sleep(6)	
		#make sure input is more than 5 words
		tmp = tmp[tmp.word_cnt>3.0]
		seed_text = ' eof '.join(list(tmp['text'].values[-3:]))
		seed_text = self.process_input_txt(seed_text)
		print('seed_text: ',seed_text)
		next_words = 100
		result = ""
		for _ in range(next_words):
			token_list = self.tokenizer.texts_to_sequences([seed_text])[0]
			token_list = pad_sequences([token_list], maxlen=self.max_sequence_len-1, padding='pre')
			predicted = np.argmax(self.model(tf.cast(token_list,tf.float32)), axis=-1)
# 			predicted_probs = self.model(tf.cast(token_list,tf.float32))[0].numpy()
# 			predicted_probs = predicted_probs/predicted_probs.sum()
# 			predicted = np.random.choice([x for x in range(len(predicted_probs))],
# 												   p=predicted_probs)
			
			output_word = ""
			for word, index in self.tokenizer.word_index.items():
				if index == predicted:
					output_word = word
					seed_text += " " + output_word
					result += " "+output_word
					break

			
		print('result: ',result)
		result = [x.replace('eof','').strip() for x in result.split(' eof ') if x.replace('eof','').strip()!='']
		#capitalize I
		result = re.sub(r'\b(i)\b','I',result[0])
		#adjust puntucation
		result = re.sub(r'([a-zA-Z])\s+([\.,!])',r'\1\2',result)
		#modify item text
		#capitalize 1st letter
		result = result[0].upper() + result[1:]
		msg = self.msg.format(**{'TEXT':result})
		exec_applescript(msg)
		
	def train(self):
		cd = CollectData()
		tmp = cd.savelogs()

		def tokenize_corpus(corpus, num_words=-1):
		  # Fit a Tokenizer on the corpus
			if num_words > -1:
				tokenizer = Tokenizer(num_words=num_words)
			else:
				tokenizer = Tokenizer()
			tokenizer.fit_on_texts(corpus)
			return tokenizer

		def create_sentences(dataset,field):
			# Remove all other punctuation
			dataset[field] = dataset[field].str.replace('[{}]'.format(string.punctuation), '')
			# Make it lowercase
			dataset[field] = dataset[field].str.lower().apply(lambda x: x+'\n')
			# Make it one long string to split by line
			lyrics = dataset[field].str.cat()
			corpus = lyrics.split('\n')
			return corpus

		def create_corpus(dataset, field):
			# Remove all other punctuation
			dataset[field] = dataset[field].str.replace('[{}]»»'.format(string.punctuation), '')
			# Make it lowercase
			dataset[field] = dataset[field].str.lower().apply(lambda x: x.replace('\n',' ')+' eof\n')
			# Make it one long string to split by line
			lyrics = dataset[field].str.cat()
			corpus = lyrics.split('\n')
			# Remove any trailing whitespace
			for l in range(len(corpus)):
				corpus[l] = corpus[l].strip()
				corpus[l] = self.process_input_txt(corpus[l])
			# Remove any empty lines
			corpus = [l for l in corpus if l != '']
	
			return corpus

		def tokenize_corpus(corpus, num_words=-1):
		  # Fit a Tokenizer on the corpus
			if num_words > -1:
				tokenizer = Tokenizer(num_words=num_words)
			else:
				tokenizer = Tokenizer()
			tokenizer.fit_on_texts(corpus)
			return tokenizer

		# Read the dataset from csv - this time with 250 songs
		dataset = tmp.copy()
		dataset['date'] = dataset['date'].apply(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))
		dataset.sort_values(by='date')
		# Create the corpus using the 'text' column containing lyrics
		corpus = create_corpus(dataset, 'text')
		# Tokenize the corpus
		tokenizer = tokenize_corpus(corpus, num_words=3500)
		total_words = tokenizer.num_words

		timestamps = dataset['date'].tolist()
		
		count=0
		sequences = []
		for i in range(len(corpus)-5):
			for j in range(1,6):
		
				if (timestamps[i+j]-timestamps[i]).seconds>60:
					continue
		
				line_in = corpus[i]
				line_out = corpus[i+j]
				start_N = len(tokenizer.texts_to_sequences([line_in]))
				line = line_in+' '+line_out
				token_list = tokenizer.texts_to_sequences([line])[0]
				for k in range(start_N, len(token_list)):
					n_gram_sequence = token_list[:k+1]
					sequences.append(n_gram_sequence)
					count+=1
			
			
		# Pad sequences for equal input length 
		max_sequence_len = max([len(seq) for seq in sequences])
		sequences = np.array(pad_sequences(sequences, maxlen=max_sequence_len, padding='pre'))

		# Split sequences between the "input" sequence and "output" predicted word
		input_sequences, labels = sequences[:,:-1], sequences[:,-1]
		# One-hot encode the labels
		one_hot_labels = tf.keras.utils.to_categorical(labels, num_classes=total_words)


		model = Sequential()
		model.add(Embedding(total_words, 128, input_length=max_sequence_len-1))
		model.add(Bidirectional(LSTM(30)))
		model.add(Dense(total_words, activation='softmax'))
		model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
		history = model.fit(input_sequences, one_hot_labels, epochs=105, verbose=1, batch_size=100)
		tf.saved_model.save(model,'model')

		# saving
		with open('tokenizer.pickle', 'wb') as handle:
			pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
	
		with open('max_sequence_len.pickle', 'wb') as handle:
			pickle.dump(max_sequence_len, handle, protocol=pickle.HIGHEST_PROTOCOL)
			
		self.reload_model()
		
	def process_input_txt(self,x):
		x = x.replace('\n',' ')
		x = re.sub(r'([a-zA-Z])([\.,!])',r'\1 \2',x)
		x = re.sub(r'(\|.*?\|h\[)(.*?)(]\|h\|r)',r' \2 ',x)
		return x