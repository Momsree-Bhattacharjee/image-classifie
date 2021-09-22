#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing dependancies
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# using pandas to read the database stored in the same folder
data = pd.read_csv("mnist.csv")


# In[2]:


# viewing column heads
data.head()


# In[4]:


#extracting data from the dataset and viewing them up close
a = data.iloc[3,1:].values


# In[5]:


# reshaping the extracted data into a reasonable size
a = a.reshape(28,28).astype('uint.28')
plt.inshow(a)


# In[6]:


#preparing the data
#separatinh labels and data values
df_x = data.iloc[:,:1]
df_y = data.iloc[:,0]


# In[7]:


#creating test and train sizes/batches
x_train , y_train , x_test , y-test= train_test_split(df_x, df_y, test_size=0.2 , random_state=4)


# In[8]:


#check data
y_train.head()


# In[9]:


#call rf classifier
rf = RandomForestClassifier(n_estimators=100)


# In[10]:


#fit the model
pred = rf.predict(x_test)


# In[11]:


pred


# In[12]:


#check prediction accuracy
s = y_test.values

#calculate the number of predicted values
count = 0
for i in range (len(pred)):
    if pred(i) == s(i):
        count = count +1


# In[13]:


count


# In[14]:


# total value that the prediction code was run on
len(pred)


# In[15]:


#accuracy value
8090/8400


# In[ ]:




