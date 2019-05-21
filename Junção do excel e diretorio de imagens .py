#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import os


# In[3]:


# Prepare 1st table (artworks-arquivo excel)

artworks = pd.read_csv(r'C:\Users\annes\OneDrive\Bureau\data\artists.csv')
artworks.dataframeName = 'artists.csv'
nRow, nCol = artworks.shape
print(f'There are {nRow} rows and {nCol} columns')


# In[4]:


artworks.head()


# In[6]:


artworks_shrunk= artworks.drop(labels=['id','years','nationality','bio','wikipedia','paintings'],axis=1)
artworks_shrunk.head(2)


# In[9]:


artworks_shrunk[['genre_1','genre_2','genre_3']] = artworks_shrunk['genre'].str.split(",",expand=True)
artworks_shrunk


# In[18]:


artworks_final=artworks_shrunk.drop('genre',1)
artworks_final


# In[11]:


# Prepare 2nd table 'df'
directory = r"C:\Users\annes\OneDrive\Bureau\data\resized"
data = []
for file in sorted(os.listdir(directory)):
    data.append(file)


# In[12]:


df = pd.DataFrame(columns=['File_complete_name'])
df['File_complete_name']=data
df.head(2)


# In[13]:


# new data frame with split value columns 
new=df['File_complete_name'].str.rsplit("_", n = 1,expand= True) 
new_b=new[1].str.rsplit(".", n = 1,expand= True) 

df['artist_name'] = new[0]
df['picture_nb'] = new_b[0]

df['artist_name'] = df['artist_name'].str.replace(r'_', ' ')
df.head(2)


# In[21]:


# Merge the two tables artworks_final and df
df_final=df.merge(artworks_final,left_on=df.artist_name, right_on=artworks_final.name)
df_final.drop(['key_0','name'],1,inplace=True)
df_final


# In[23]:


df_final.shape

