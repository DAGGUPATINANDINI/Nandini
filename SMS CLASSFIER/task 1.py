#!/usr/bin/env python
# coding: utf-8

# # Importing the packages and libraries

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# # Reading and exploring the data

# In[3]:


# Reading the data
data= pd.read_csv("C:\\Users\\daggu\\Downloads\\spam.csv",encoding='latin-1')
print(data)


# In[4]:


data.describe()


# In[5]:


data.sample(7)


# In[6]:


data.shape


# In[7]:


data.count()


# In[8]:


data = data.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)
data = data.rename(columns={'v1':'label','v2':'Text'})
data['label_enc'] = data['label'].map({'ham':0,'spam':1})
data.head()


# In[9]:


sns.countplot(x=data['label'])
plt.show()


# In[10]:


# Find average number of tokens in all sentences
avg_words_len=round(sum([len(i.split()) for i in data['Text']])/len(data['Text']))
print(avg_words_len)


# In[11]:


# Finding Total no of unique words in corpus
s = set()
for sent in data['Text']:
  for word in sent.split():
    s.add(word)
total_words_length=len(s)
print(total_words_length)


# In[12]:


# ham messages
data[data['label'] == 0].describe()


# In[13]:


sns.pairplot(data, hue = 'label')


# In[14]:


import matplotlib.pyplot as plt
plt.pie(data['label'].value_counts(),labels = ['hum','spam'],autopct ="%0.1f")
plt.show()


# In[ ]:





# In[ ]:




