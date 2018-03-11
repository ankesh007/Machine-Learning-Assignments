
# coding: utf-8

# In[1]:


from nltk.tokenize import RegexpTokenizer
import sys
import re


# In[3]:


tokenizer = RegexpTokenizer(r'\w+')


# In[17]:


def getCleanedData(inputFileName,outputFileName):
    out = open(outputFileName, 'w')
    cleaned_list=[]
    with open(inputFileName) as f:
        docs = f.readlines()
    for doc in docs:
        raw = doc.lower()
        raw = raw.replace("<br /><br />", " ")
#         print raw
        raw = re.sub(r"[^a-z0-9]+"," ",raw)
        raw = tokenizer.tokenize(raw)
        cleaned_list.append(raw)
        documentWords = ' '.join(raw)
        # print >> out , documentWords # for python 2.x
        print((documentWords), file=out) #for python 3.x 
    out.close()
    return cleaned_list


# In[20]:


# cleaned_list=getCleanedData('Dataset/imdb/imdb_test_text.txt','Dataset/Clean/test_text.txt')
cleaned_list=getCleanedData(sys.argv[1],sys.argv[2])

# print len(cleaned_list)
# print cleaned_list


# In[ ]:




