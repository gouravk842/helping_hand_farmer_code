#!/usr/bin/env python
# coding: utf-8

# In[2]:


from keras.applications import VGG16


# In[3]:


model = VGG16(weights = 'imagenet', 
                 include_top = False, 
                 input_shape = (224, 224, 3))


# In[4]:


for layer in model.layers:
    layer.trainable = False


# In[5]:


def addTopModel(bottom_model, num_classes, D=256):
    """creates the top or head of the model that will be 
    placed ontop of the bottom layers"""
    top_model = bottom_model.output
    top_model = Flatten(name = "flatten")(top_model)
    top_model = Dense(D, activation = "relu")(top_model)
    top_model = Dropout(0.3)(top_model)
    top_model = Dense(num_classes, activation = "softmax")(top_model)
    return top_model


# In[6]:


model.input


# In[7]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model


# In[8]:


num_classes = 10


# In[9]:


FC_Head = addTopModel(model, num_classes)
modelnew = Model(inputs=model.input, outputs=FC_Head)
print(modelnew.summary())


# In[10]:


from keras.preprocessing.image import ImageDataGenerator


# In[11]:


train_data_dir = "C:/Users/Mon_Amour/Desktop/dataset_project/Mango Leaf Species (train)"
validation_data_dir = "C:/Users/Mon_Amour/Desktop/dataset_project/Mango Leaf Species (test)"


# In[12]:


train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=20,
      width_shift_range=0.2,
      height_shift_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')


# In[13]:


validation_datagen = ImageDataGenerator(rescale=1./255)


# In[14]:


train_batchsize = 16
val_batchsize = 10


# In[15]:


train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(224, 224),
        batch_size=train_batchsize,
        class_mode=('categorical'))


# In[16]:


validation_generator = validation_datagen.flow_from_directory(
                validation_data_dir,
                target_size=(224, 224),
                batch_size=val_batchsize,
                class_mode='categorical',
                shuffle = False,)


# In[17]:


from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping


# In[18]:


checkpoint = ModelCheckpoint("C:/Users/Mon_Amour/Desktop/dataset_project/mango_detection.h5",
                             monitor="val_loss",
                             mode="min",
                             save_best_only = True,
                             verbose=1)

earlystop = EarlyStopping(monitor = 'val_loss', 
                          min_delta = 0, 
                          patience = 3,
                          verbose = 1,
                          restore_best_weights = True)


# In[19]:


callbacks = [earlystop, checkpoint]


# In[20]:


modelnew.compile(loss = 'categorical_crossentropy',
              optimizer = RMSprop(lr = 0.001),
              metrics = ['accuracy'])


# In[21]:


nb_train_samples = 300
nb_validation_samples = 40
epochs = 5
batch_size = 3


# In[21]:


history = modelnew.fit_generator(
    train_generator,
    steps_per_epoch = nb_train_samples // batch_size,
    epochs = epochs,
    callbacks = callbacks,
    validation_data = validation_generator,
    validation_steps = nb_validation_samples // batch_size)


# In[22]:


train_generator.class_indices


# In[ ]:





# In[ ]:




