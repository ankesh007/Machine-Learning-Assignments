
# coding: utf-8

# In[1]:


from nltk.tokenize import RegexpTokenizer
import sys
import re
import numpy as np
import pandas as pd
import math
import random


# In[2]:


tokenizer = RegexpTokenizer(r'\w+')


# In[3]:


def tokenizedInput(inputFileName,tokenize=False):
    cleaned_list=[]
    with open(inputFileName) as f:
        docs = f.readlines()

    for doc in docs:
        raw=None
        if(tokenize==True):
            raw = tokenizer.tokenize(doc)
        else:
            raw=doc
        cleaned_list.append(raw)
    
    return cleaned_list


# In[4]:


# x_train=tokenizedInput('Dataset/Clean1/train_text.txt',tokenize=True)


# In[5]:


# print(len(x_train))
# print(len(x_train[0]))


# In[6]:


class NaiveBayes:
    
    def __init__(self,x,y,laplace_smoother=1):
        self.x=list(x)
        self.y=list(y)
        self.laplace_smoother=laplace_smoother
        self.distinct_x=0
        self.distinct_y=0
        self.instances=len(y)
        self.x_dict={}
        self.vocabulary={}
        self.y_dict={}
        self.revMapy={}
        self.wordsInClass=[]
        self.instancesInClass=[]
    
    def calculateParameters(self):
#         classes=np.unique(self.npy)
        counter=0
        for i in range(self.instances):
            feature_vector=self.x[i]
            feature_class=self.y[i]
            mapped_class=counter
            
            if(feature_class in self.y_dict):
                mapped_class=self.y_dict[feature_class]
            else:
                self.revMapy[counter]=feature_class
                self.y_dict[feature_class]=counter
                self.wordsInClass.append(0)
                self.instancesInClass.append(0)
                counter+=1
            
            self.instancesInClass[mapped_class]+=1
            
            for words in feature_vector:
                if(words in self.vocabulary):
                    pass
                else:
                    self.vocabulary[words]=1
                
                key=(words,mapped_class)
                self.wordsInClass[mapped_class]+=1
                
                if(key in self.x_dict):
                    self.x_dict[key]+=1
                else:
                    self.x_dict[key]=1
            
        self.distinct_x=len(self.vocabulary)
        self.distinct_y=len(self.y_dict)
#         self.printParameters()
    
    def getLogPrior(self,label):
        return math.log(float(self.instancesInClass[label])/self.instances)
    
    def getLogProb(self,attribute,label):
        
        occurences=0
        key=(attribute,label)
        
        if key in self.x_dict:
            occurences=self.x_dict[key]
        
        occurences+=self.laplace_smoother
        
        total_occurences_in_class=self.wordsInClass[label]
        total_occurences_in_class+=self.distinct_x*self.laplace_smoother
        
        return math.log(float(occurences)/total_occurences_in_class)
        
    def getClass(self,x):
        max_log_prob=-1e9
        label=-1
        
        for i in range(self.distinct_y):
            log_prob_x_given_y=0
            
            for attributes in x:
                log_prob_x_given_y+=self.getLogProb(attributes,i)
            
            log_prob_x=log_prob_x_given_y+self.getLogPrior(i)
            
            if(log_prob_x>max_log_prob):
                max_log_prob=log_prob_x
                label=self.revMapy[i]
            
        return label
    
    def ConfusionMatrix(self,y,predy):
#         confusion=[[0]*self.distinct_y]*self.distinct_y
        confusion = [ [0]*self.distinct_y for _ in range(self.distinct_y) ]
        
        tests=len(y)
        
        for i in range(tests):
            confusion[self.y_dict[predy[i]]][self.y_dict[y[i]]]+=1
        
        for i in range(self.distinct_y):
            for j in range(self.distinct_y):
#                 print((i,j),end=' ')                                
                print("%5d"%(confusion[i][j]),end=' ')
            print()
        
        
    def getAccuracy(self,x,y,printConfusionMatrix=False):
        total_tests=len(y)
        passed_tests=0
        prediction_list=[]
        
        for i in range(total_tests):
            xi=x[i]
            yi=y[i]
            
#             if y[i] in self.y_dict:
#                 pass
#             else:
#                 continue
            pred_yi=self.getClass(xi)
            prediction_list.append(pred_yi)
            if(pred_yi==yi):
                passed_tests+=1
        
        if(printConfusionMatrix==True):
            self.ConfusionMatrix(y,pred_yi)
                
        return [prediction_list,(float(passed_tests))/total_tests]
    
    def getAccuracyRandomPredict(self,x,y):
        
        average_over=10
        total_accuracy=0
        
        for i in range(average_over):
            
            total_tests=len(y)
            passed_tests=0

            for i in range(total_tests):
                yi=y[i]

                pred_yi=random.randint(0,self.distinct_y-1)
                pred_yi=self.revMapy[pred_yi]

                if(pred_yi==yi):
                    passed_tests+=1

            total_accuracy+=(float(passed_tests))/total_tests
        
        return total_accuracy/average_over
    
    def getAccuracyMajorityPredictor(self,x,y):
        max_occ=-1
        majority_class=-1
        total_tests=len(y)
        passed_tests=0        
        
        for i in range(self.distinct_y):
            if(max_occ<self.instancesInClass[i]):
                max_occ=self.instancesInClass[i]
                majority_class=i
        
        if(majority_class==-1):
            return 0
        
        majority_class=self.revMapy[majority_class]
        
        for yi in y:
            if(yi==majority_class):
                passed_tests+=1
                
        return (float(passed_tests))/total_tests       

    def printParameters(self):
        print("Vocabulary Size:",self.distinct_x)
        print("Classes:",self.distinct_y)
        print("X_Y's:",len(self.x_dict))
        
        print("Mapped Classes:", self.y_dict)
        
        print("Instances In Mapped class:",end='')
        
        for counts in self.instancesInClass:
            print(counts,end=' ')
        
        print("")
        print("Words In Mapped class:",end='')
        
        for counts in self.wordsInClass:
            print(counts,end=' ')
        
        print("")
        


# In[46]:


x_train=tokenizedInput('Dataset/imdb/imdb_train_text.txt',tokenize=True)
y_train=tokenizedInput('Dataset/imdb/imdb_train_labels.txt',tokenize=False)
y_train=list(map(int,y_train))


# In[47]:


# len(x_train)


# In[48]:


# len(y_train)


# In[49]:


# type(y_train[0])


# In[50]:


NaiveBayesClassifier=NaiveBayes(x=x_train,y=y_train)


# In[51]:


NaiveBayesClassifier.calculateParameters()


# In[52]:


print (sum(NaiveBayesClassifier.wordsInClass))


# In[53]:


# print (NaiveBayesClassifier.getAccuracy(x_train,y_train))[1]


# In[54]:


x_test=tokenizedInput('Dataset/imdb/imdb_test_text.txt',tokenize=True)
y_test=tokenizedInput('Dataset/imdb/imdb_test_labels.txt',tokenize=False)
y_test=list(map(int,y_test))


# In[57]:


[a,b]=(NaiveBayesClassifier.getAccuracy(x_test,y_test,printConfusionMatrix=False))
print (b)
# print(NaiveBayesClassifier.ConfusionMatrix(a,y_test))


# In[ ]:


# print (NaiveBayesClassifier.getAccuracyRandomPredict(x_train,y_train))


# In[ ]:


# print (NaiveBayesClassifier.getAccuracyMajorityPredictor(x_test,y_test))


# In[35]:





# In[58]:


NaiveBayesClassifier.ConfusionMatrix(y_test,a)


# In[251]:


NaiveBayesClassifier.y_dict

