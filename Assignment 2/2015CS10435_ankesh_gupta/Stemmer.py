
# coding: utf-8

# In[4]:


from nltk.tokenize import RegexpTokenizer
import sys
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords


# In[5]:


tokenizer = RegexpTokenizer(r'\w+')
en_stop = set(stopwords.words('english'))
p_stemmer = PorterStemmer()


# In[7]:


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
        tokens = tokenizer.tokenize(raw)        
        stopped_tokens = [token for token in tokens if token not in en_stop]
        stemmed_tokens = [p_stemmer.stem(token) for token in stopped_tokens]
        documentWords = ' '.join(stemmed_tokens)
#         print((documentWords), file=out)

        # print >> out , documentWords # for python 2.x
        print((documentWords), file=out) #for python 3.x 
    out.close()
    return cleaned_list


# In[9]:


# cleaned_list=getCleanedData('Dataset/imdb/imdb_test_text.txt','Dataset/Clean/test_text.txt')
cleaned_list=getCleanedData(sys.argv[1],sys.argv[2])

# print len(cleaned_list)
# print cleaned_list


# In[ ]:




