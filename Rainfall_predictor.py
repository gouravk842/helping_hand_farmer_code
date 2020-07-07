#!/usr/bin/env python
# coding: utf-8

# In[23]:


import joblib


# In[24]:


model = joblib.load('C:/Users/Mon_Amour/Desktop/dataset_project/rainfall_model.pk1')


# In[25]:


n=int(input("enter the year "))


# In[26]:


cm=model.predict([[n]])


# In[27]:


print(f" Expected rainfall in the year {n} is {cm[0]}")


# In[ ]:




