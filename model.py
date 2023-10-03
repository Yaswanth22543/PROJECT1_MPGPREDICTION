#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


import os
os.getcwd()


# In[3]:


mpg_df = pd.read_csv("Auto MPG Reg.csv")


# In[4]:


mpg_df.horsepower = mpg_df.horsepower.fillna(mpg_df.horsepower.median())


# In[5]:


y = mpg_df.mpg
X = mpg_df.drop('mpg',axis = 1)


# In[6]:


from sklearn.linear_model import LinearRegression


# In[7]:


reg= LinearRegression()


# In[8]:


reg.fit(X,y)


# In[9]:


reg.score(X,y)


# In[10]:


import joblib


# In[11]:


joblib.dump(reg,"reg_model.sav")


# In[ ]:




