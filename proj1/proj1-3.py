#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import matplotlib.pyplot as plt


# In[3]:


import numpy as np


# In[4]:


import statsmodels.api as sm


# In[5]:


DF = 'traceX.txt'


# In[6]:


X = np.loadtxt(DF)


# In[7]:


avgx = (X.mean(axis = 0))


# In[8]:


plt.plot(avgx)


# In[9]:


DFY = 'traceY.txt'


# In[10]:


Y = np.loadtxt(DFY)


# In[11]:


avgy = (Y.mean(axis = 0))


# In[12]:


plt.plot(avgy)


# In[34]:


X_acov = sm.tsa.stattools.acovf(avgx, fft = False, nlag = 30)


# In[35]:


plt.plot(X_acov)


# In[19]:


Y_acov = sm.tsa.stattools.acovf(avgy, fft = True)


# In[20]:


plt.plot(Y_acov)


# In[24]:


print(Y_acov)


# In[21]:


sm.graphics.tsa.plot_acf(avgx)


# In[23]:


sm.graphics.tsa.plot_acf(avgy)


# In[32]:





# In[31]:





# In[ ]:




