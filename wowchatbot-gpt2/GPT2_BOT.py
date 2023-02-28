from transformers import GPT2LMHeadModel, GPT2Tokenizer
import random
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
import torch.nn.functional as F
import string
import numpy as np
import time
from subprocess import Popen, PIPE
from Dataset import CollectData
import pickle
import re
import datetime
import collections
import time
import os

def exec_applescript(script):
	p = Popen(['osascript', '-e', script])


class ChatBotText():
	
	def create_corpus(self,dataset, field):
		# Remove all other punctuation
		dataset[field] = dataset[field].str.replace('[{}]»»/'.format(string.punctuation), '')
		# Make it lowercase
		dataset[field] = dataset[field].str.lower().apply(lambda x: x.replace('\n',' ')+' \n')
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


	def process_input_txt(self,x):
		x = x.replace('\n',' ')
		x = re.sub(r'([a-zA-Z])([\.,!?])',r'\1 \2',x)
		x = re.sub(r'([\.,!?])([a-zA-Z])',r'\1 \2',x)
		

		
		elephant_tags = re.findall('(\|.*?\|h\|r)',x)
		for et in elephant_tags:
			print(et)
			et_text = re.findall(r'(?:h\[)(.*?)[\||\]]', et)
			print(et_text)
			if len(et_text)>0:
				x = x.replace(et, et_text[0])
	
		return x


class TorchChatLogs(ChatBotText):
	
	def __init__(self, inputs_outputs, tokenizer, truncate=False, gpt2_type="gpt2", max_length=1024):
		
		self.chat_count = 0
		self.entries = []
		for (line_in, line_out) in inputs_outputs:

			self.entries.append(torch.tensor( tokenizer.encode(
				f"<|startoftext|>Input: {line_in}\nOutput: {line_out[:max_length]}<|endoftext|>"
															   )
											 ))
			self.chat_count+=1

	def __len__(self):
		return self.chat_count

	def __getitem__(self, item):
		return self.entries[item]


class ChatBot(ChatBotText):
	
	def __init__(self):
		
#		self.model = tf.keras.models.load_model('./keras_model.h5')
#		with open('tokenizer.pickle', 'rb') as handle:
#			self.tokenizer = pickle.load(handle)
#
#		with open('max_sequence_len.pickle', 'rb') as handle:
#			self.max_sequence_len = pickle.load(handle)
#
		#Get the tokenizer and model
		self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token = '<|startoftext|>',
													   eos_token = '<|endoftext|>')
		self.model = GPT2LMHeadModel.from_pretrained('gpt2')
		self.model.resize_token_embeddings(len(self.tokenizer))
		
		self.model_version_num = 0

		# Apple Script function to send text message to chat
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
	
	def reload_model(self, epoch_num = None):
	
		MODEL_PATH = 'gpt2-models'

		if epoch_num is None:
			files = []
			for _, _, f in os.walk('gpt2-models'):
				files.extend(f)
		
			max_version = 0
			for f in files:
				v = re.findall('-(\d+).',f)
				if len(v)>0:
					v = int(v[0])
					if v > max_version:
						max_version = v

		if max_version > 0:
				self.model_version_num = max_version
				PATH = 'gpt2-models/wowchatbot-{}.pt'.format(max_version)
				self.model.load_state_dict(torch.load(PATH))

		else:

			PATH = 'gpt2-models/wowchatbot-{}.pt'.format(epoch_num)
			self.model.load_state_dict(torch.load(PATH))

	def create_input_outputs(self):
		
		cd = CollectData()
		result = []
		dataset = cd.collectlogs().copy()
		dataset['date'] = dataset['date'].apply(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))
		dataset.sort_values(by='date',inplace=True)
		# Create the corpus using the 'text' column
		corpus = self.create_corpus(dataset, 'text')
		timestamps = dataset['date'].tolist()
	
		self.chat_count=0
		sequences = []
		for i in range(len(corpus)-6):
			for j in range(1,7):
				
				if (timestamps[i+j]-timestamps[i]).seconds>100:
					continue
						
				line_in = corpus[i]
				line_out = corpus[i+j]

				result.append((line_in, line_out))

		return result

	def generate_response(
						  self,
						  model,
						  tokenizer,
						  prompt,
						  entry_count=10,
						  entry_length=100, #maximum number of words
						  top_p=0.8,
						  temperature=1.):
						  
		
		model.eval()
		generated_num = 0
		generated_list = []
		
		filter_value = -float("Inf")
		
		unprocessed_prompt = prompt
		prompt = self.process_input_txt(prompt)
		prompt = f"<|startoftext|>Input: {prompt}\nOutput:"
		print(prompt)
		with torch.no_grad():
			
			for entry_idx in range(entry_count):
				
				entry_finished = False
				generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
				print(generated)
				for i in range(entry_length):
					outputs = model(generated, labels=generated)
					loss, logits = outputs[:2]
					logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
					outputs = model(generated, labels=generated)
					
					loss, logits = outputs[:2]
					logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)

					sorted_logits, sorted_indices = torch.sort(logits, descending=True)
					cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

					sorted_indices_to_remove = cumulative_probs > top_p
					sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1
																					 ].clone()
					sorted_indices_to_remove[..., 0] = 0

					indices_to_remove = sorted_indices[sorted_indices_to_remove]
					logits[:, indices_to_remove] = filter_value

					next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
					generated = torch.cat((generated, next_token), dim=1)

					if next_token in tokenizer.encode("<|endoftext|>"):
						entry_finished = True
					
					if entry_finished:
						
						generated_num = generated_num + 1

						output_list = list(generated.squeeze().numpy())
						output_text = tokenizer.decode(output_list)
						generated_list.append(output_text)
						break

				if not entry_finished:
					output_list = list(generated.squeeze().numpy())
					output_text = f"{tokenizer.decode(output_list)}<|endoftext|>"
					generated_list.append(output_text)



		result = generated_list[0]
		if len(result)==0:
			print('no result')
			return 0
		
#		result = result.replace(prompt,'').replace('<|endoftext|>', '').strip()
#
#
#		result = re.sub(r'([a-zA-Z])\s+([\.,!?\n])',r'\1\2',result)
#		result = re.findall(r'([a-zA-Z][^\.!?]*[\.!?])',result)
#		result = ' '.join(result)

		return result

	def execute(self):
		cd = CollectData()
		tmp = cd.collectchat()
		print('collecting data')
		time.sleep(1)
		#make sure input is more than 5 words
		tmp = tmp[tmp.word_cnt>3.0]
		seed_text = tmp['text'].values[-1]

		print(seed_text)
		result = self.generate_response(
										self.model.to('cpu'),
										self.tokenizer,
										seed_text,
										entry_count=1)
		print(result)
		result = result.replace('<|startoftext|>', '').replace('<|endoftext|>', '').strip()
		result = ' '.join(re.findall('Output\:(.+?)(?:\n|$|Output\:)', result))
		print(result)
		msg = self.msg.format(**{'TEXT':result})
		exec_applescript(msg)

	def train(self,
			  batch_size=16, epochs=5, lr=2e-5,
			  max_seq_len=400, warmup_steps=200,
			  gpt2_type="gpt2", output_dir="gpt2-models", output_prefix="wowchatbot",
			  test_mode=False,save_model_on_epoch=True):
		
		inputs_outputs = self.semisuper_filter()
		dataset = TorchChatLogs(inputs_outputs, self.tokenizer, truncate=True, gpt2_type="gpt2")

		#Accumulated batch size (since GPT2 is so big)
		def pack_tensor(new_tensor, packed_tensor, max_seq_len):
			if packed_tensor is None:
				return new_tensor, True, None
			if new_tensor.size()[1] + packed_tensor.size()[1] > max_seq_len:
				return packed_tensor, False, new_tensor
			else:
				packed_tensor = torch.cat([new_tensor, packed_tensor[:, 1:]], dim=1)
				return packed_tensor, True, None

		acc_steps = 100
		device=torch.device("cpu")
		self.model.train()

		optimizer = AdamW(self.model.parameters(), lr=lr)
		scheduler = get_linear_schedule_with_warmup(
													optimizer, num_warmup_steps=warmup_steps, num_training_steps=-1
													)
	
		train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
		loss=0
		accumulating_batch_count = 0
		input_tensor = None
		
		for epoch in range(epochs):
			
			print(f"Training epoch {epoch}")
			print(loss)
			for idx, entry in enumerate(train_dataloader):
				(input_tensor, carry_on, remainder) = pack_tensor(entry, input_tensor, 768)
				
				if carry_on and idx != len(train_dataloader) - 1:
					continue
				
				input_tensor = input_tensor.to(device)
				outputs = self.model(input_tensor, labels=input_tensor)
				loss = outputs[0]
				loss.backward()
			
				if (accumulating_batch_count % batch_size) == 0:
					optimizer.step()
					scheduler.step()
					optimizer.zero_grad()
					self.model.zero_grad()
				
				accumulating_batch_count += 1
				input_tensor = None

			if save_model_on_epoch:
				torch.save(
						   self.model.state_dict(),
						   os.path.join(output_dir, f"{output_prefix}-{epoch+self.model_version_num+1}.pt"),
						   )

	def semisuper_filter(self):
	
		inputs_outputs = self.create_input_outputs()
		dataset = TorchChatLogs(inputs_outputs, self.tokenizer, truncate=True, gpt2_type="gpt2")

		train_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
		
		losses = []
		device=torch.device("cpu")
		for idx, entry in enumerate(train_dataloader):
			
			input_tensor = entry.to(device)
			outputs = self.model(input_tensor, labels=input_tensor)
			loss = outputs[0]
			losses.append(loss.detach().numpy())

		losses = np.array(list(losses)).flatten()
		filter_indx = np.where(losses<=np.percentile(losses,20))[0]
		return [row for i, row in enumerate(inputs_outputs) if i in filter_indx]
			
	def semisuper_train(self,epochs=100):
		
		dataset,corpus,cdict,scores = self.semisuper_filter()



