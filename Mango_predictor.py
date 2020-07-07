#!/usr/bin/env python
# coding: utf-8

# In[35]:


from keras.models import load_model
import cv2
import numpy as np


# In[36]:


model = load_model('C:/Users/Mon_Amour/Desktop/dataset_project/mango_detection.h5')


# In[37]:


image_path=input("enter the image path")


# In[38]:


input_im = cv2.imread(image_path)
input_im = cv2.resize(input_im, (224, 224), interpolation = cv2.INTER_LINEAR)
input_im = input_im / 255.
input_im = input_im.reshape(1,224,224,3)

res = np.argmax(model.predict(input_im, 1, verbose = 0), axis=1)


# In[39]:


if res == 0:
    print("Alphanso")
elif res == 1:
    print("Amarpali")
elif res == 2:
    print("Ambika")
elif res == 3:
    print("Austin")
elif res == 4:
    print("Chausa Desi")
elif res == 5:
    print("Dasheri Desi")
elif res == 6:
    print("Duthpedha")
elif res == 7:
    print("Farnadeen")
elif res == 8:
    print("Keshar")
elif res == 9:
    print("Langra Desi")


# In[ ]:




