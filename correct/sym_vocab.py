#!/usr/bin/env python
# coding: utf-8

# In[4]:


from __future__ import absolute_import, division, print_function, unicode_literals
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer


# In[5]:


# import tensorflow as tf
import nltk, re, pprint
import docx2txt
import numpy as np
import os

# download 
nltk.download('words')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# In[47]:


def crawling_dict(word):
    CSAT_dict = {}
    syn_list = []
    basic_url = "https://endic.naver.com/search.nhn?sLn=kr&searchOption=thesaurus&query={}".format(word)
    
    try:
        req = Request(basic_url)
        res = urlopen(req)
        html = res.read().decode('utf-8')
        
        bs = BeautifulSoup(html, 'html.parser')
        tags = bs.findAll('a', {'class': 'N=a:tsr.entry'})
        
        for tag in tags:
            syn = tag.get_text()
            syn_list.append(syn)   
            
        CSAT_dict[word] = syn_list[:5]
        
        return CAST_dict
        
        
    except:
        # print('{}는 사전에 없는 단어입니다'.format(word))
        
        CSAT_dict[word] = word
        
        return CSAT_dict

    



# In[ ]:





# In[ ]:




