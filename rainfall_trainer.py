#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


dataset = pd.read_csv("C:/Users/Mon_Amour/Desktop/dataset_project/bihar_rainfall.csv")


# In[3]:


dataset


# In[4]:


y = dataset['rainfall']


# In[5]:


x=dataset['year']


# In[6]:


x=x.values


# In[7]:


z=x.shape


# In[8]:


X=x.reshape(z[0],1)


# In[9]:


import cv2


# In[10]:


from sklearn.linear_model import LinearRegression


# In[11]:


mind = LinearRegression()


# In[12]:


mind.fit(X,y)


# In[2]:


import joblib


# In[18]:


joblib.dump(mind,'C:/Users/Mon_Amour/Desktop/dataset_project/rainfall_model.pk1')


# In[ ]:




