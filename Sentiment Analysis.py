
# coding: utf-8

# In[130]:


import numpy as np
import pandas as pd
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *


df = pd.read_csv('C:/Users/Akanksha/Desktop/NYTIMES.csv')


# In[131]:


df["neg"] = ''
df["neu"] = ''
df["pos"] = ''
Headline=[]
Headline = df['Headline']


# In[132]:


Headline1=list(Headline)


# In[133]:


#print(Headline1)
list_a=[]
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from nltk import tokenize
sid = SentimentIntensityAnalyzer()

for H in Headline1:
    #print(sentence)
    ss = sid.polarity_scores(H)
    list_a.append(ss)
list_a
neg=[]
for a in list_a:
    neg.append(a['neg'])
    
neutral = []
for a in list_a:
    neutral.append(a['neu'])
    
positive = []
for a in list_a:
    positive.append(a['pos'])
    


# In[134]:


df["neg"] = neg
df["neu"] = neutral
df["pos"] = positive


# In[135]:


del df['Unnamed: 0']


# In[137]:


df

