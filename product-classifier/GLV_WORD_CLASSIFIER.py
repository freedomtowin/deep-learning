import numpy as np
import pandas as pd
import json
import re
import os
class GLV_WORD_CLASSIFIER():
    
    def insert_extended_vocab(self):
        print('extended vocab')
        for word in self.extnd_vocab:
            kw = np.array([self.extnd_vocab[word]])
            print(word,kw)
            self.emb_layer[word] = list(np.mean(self.get_glove_embed(kw),axis=0))



    def get_glove_embed(self, text):

        N = text.shape[0]
        embed = np.zeros((N,50))
        missings = []
        for n in range(N):
            
            tmp = text[n,0].lower()

            if tmp is None:
                missings.append(n)
                continue

            tmp = re.sub('[^a-zA-Z\-]',' ',tmp)
            grams = tmp.split(' ')
            gram_cnt = len(grams)
            for word in grams:

                if word in self.emb_layer.keys():
                    for i in range(0,50):
                        embed[n,i] = embed[n,i] + (self.emb_layer[word][i])/gram_cnt

            norm = np.sum(embed[n]**2)
            if norm==0:
                continue

            embed[n] = embed[n]/np.sqrt(norm)
        return embed



    def consine_similarity(self,v1,v2):
        return v2.dot(v1.T) / (np.linalg.norm(v1)*np.linalg.norm(v2)+1e-9)

    def categorizer(self,text):
        max_sim = 0 
        max_cat = ''
        max_subcat = ''
        result = {}
        v1 = self.get_glove_embed(text).reshape(1,-1)
        for i in self.category_embed_df.index:
            cat = self.category_embed_df.loc[i,'CATEGORY']
            subcat = self.category_embed_df.loc[i,'SUBCATEGORY']
            v2 = self.category_embed_df.loc[i,self.features].values.reshape(1,-1)
            sim = self.consine_similarity(v1,v2)

            if sim>max_sim:
                max_cat = cat
                max_subcat = subcat
                max_sim=sim
        return max_cat,max_subcat,max_sim            


    def __init__(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        f = open('embeddings\glove.6B.50d.txt','rb')
        self.emb_layer = {}
        for line in f:
            splitline = line.split()
            word = splitline[0].decode('utf-8')
            embedding = [float(val) for val in splitline[1:]]
            self.emb_layer[word] = embedding

        with open('hts_assc_cats.json','r') as file:
            self.assc_wrds = json.load(file)
            
        
        with open('extnd_vocab.json','r') as file:
            self.extnd_vocab = json.load(file)
        
        self.insert_extended_vocab()
        self.category_embed_df = pd.DataFrame(columns=('CATEGORY','SUBCATEGORY'))
        embed_lst = []
        count=0
        for cat in self.assc_wrds:
            for subcat in self.assc_wrds[cat]:
                includes = self.assc_wrds[cat][subcat]
                self.category_embed_df.loc[count] = [cat,subcat]
                assc_cat_words = np.array([[cat,subcat]+includes]).reshape(-1,1)
                embed = np.mean(self.get_glove_embed(assc_cat_words),axis=0)
                embed_lst.append(embed)
                count+=1
                
        embed_lst = np.vstack(embed_lst)
        self.features = []
        for i in range(50):
            self.category_embed_df['W'+str(i)] = embed_lst[:,i]
            self.features.append('W'+str(i))
       