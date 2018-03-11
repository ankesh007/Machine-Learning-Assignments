
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import sys

# In[3]:




# In[18]:


def correctData(pd,outputfile):
    val=pd.values
    [instances,dim]=val.shape
    zeros=np.zeros((instances,1))
    val=np.concatenate((val,zeros),axis=1)
    out=open(outputfile,'w')
    y=val[:,784:785]
    x=val[:,0:784]
    
    instances=y.shape[0]
    
    for i in range(instances):
        print (y[i,0],end="",file=out)
        for j in range(784):
            print (" "+str(j+1)+":"+str(x[i,j]),end="",file=out)
        
        print(file=out)
    
    out.close()
        


# In[20]:


trainpd=pd.read_csv(sys.argv[1],header=None,sep=',')
correctData(trainpd,sys.argv[2])

