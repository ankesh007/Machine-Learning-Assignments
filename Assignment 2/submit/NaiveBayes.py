
from nltk.tokenize import RegexpTokenizer
import sys
import re
import numpy as np
import pandas as pd
import math
import random
import pickle


# In[4]:


tokenizer = RegexpTokenizer(r'\w+')


# In[5]:


def tokenizedInput(inputFileName,tokenize=True):
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


# In[11]:


def tokenizedInput_WithNGrams(inputFileName,grams=1):
    cleaned_list=[]
    with open(inputFileName) as f:
        docs = f.readlines()

    for doc in docs:
        raw2=[]
        raw = tokenizer.tokenize(doc)
        leng=len(raw)
        for k in range(grams-1,leng,1):
            new_token=""
            for l in range(k-grams+1,k+1,1):
                new_token+=" "+raw[l]
            raw2.append(new_token)
        cleaned_list.append(raw2+raw)
    
    return cleaned_list



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
                    self.vocabulary[words]+=1
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

    def purge(self,threshold=7):
        
        emp_lis=[]
        
        for key,value in self.vocabulary.items():
            if(value<threshold and len(tokenizer.tokenize(key))>1):
                self.distinct_x-=1
                emp_lis.append(key)
                for labels in range(self.distinct_y):
                    new_key=(key,labels)
                    if(new_key in self.x_dict):
                        count=self.x_dict[new_key]
                        del self.x_dict[new_key]
                        self.wordsInClass[labels]-=count
        
        for words in emp_lis:
            del self.vocabulary[words]
    
    def getLogPrior(self,label):
        return math.log(float(self.instancesInClass[label])/self.instances)
    
    def getLogProb(self,attribute,label):
        
        occurences=0
        key=(attribute,label)
        
        if key in self.x_dict:
            occurences=self.x_dict[key]
        
        
        
#         if(occurences>10 and len(tokenizer.tokenize(attribute))>1):
#             print(attribute)
#             return 0
        
        
        occurences+=self.laplace_smoother

        total_occurences_in_class=self.wordsInClass[label]
        total_occurences_in_class+=self.distinct_x*self.laplace_smoother
        
#         if(occurences<=0 || total_occurences_in_class<=0):
            
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
        confusion = [ [0]*self.distinct_y for _ in range(self.distinct_y) ]
        
        tests=len(y)
        
        for i in range(tests):
            confusion[self.y_dict[predy[i]]][self.y_dict[y[i]]]+=1
        
        for i,ii in self.y_dict.items():
            for j,jj in self.y_dict.items():
                print("%5d"%(confusion[ii][jj]),end=' ')
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

def main():

    if(len(sys.argv)<5):
        print ("Usage: <script> <mode> <inputfilename> <input_labels> <modelName> <outputFileName>")
        print ("mode -> 0 = Train | 1 = Predict | 2 = Accuracy")
        print ("<outputFileName> is required for mode=1")
        print ("<input_labels> are required for mode=0/2")
        exit()

    mode=int(sys.argv[1])

    if(mode==0):

        modelName=sys.argv[4]
        x_train=tokenizedInput(sys.argv[2])
        y_train=tokenizedInput(sys.argv[3],tokenize=False)
        y_train=list(map(int,y_train))
        NaiveBayesClassifier=NaiveBayes(x_train,y_train)
        NaiveBayesClassifier.calculateParameters()
        f=open(modelName,"wb")
        pickle.dump(NaiveBayesClassifier,f)
        f.close()
        
    elif(mode==1):
        if(len(sys.argv)<5):
            print ("Wrong format for mode=1. Exitting")
            exit()
        modelName=sys.argv[3]
        outputFileName=sys.argv[4]

        x_test=tokenizedInput(sys.argv[2])
        lenx=len(x_test)
        y_test=[0]*lenx

        NaiveBayesClassifier=None
        with open(modelName, "rb") as f:
            NaiveBayesClassifier = pickle.load(f)    

        [predictions,accu]=NaiveBayesClassifier.getAccuracy(x_test,y_test)
        np.savetxt(outputFileName,predictions,fmt="%d")

    else:
        if(len(sys.argv)<5):
            print ("Wrong format for mode=1. Exitting")
            exit()

        modelName=sys.argv[4]
        x_test=tokenizedInput(sys.argv[2])
        y_test=tokenizedInput(sys.argv[3],tokenize=False)
        y_test=list(map(int,y_test))

        NaiveBayesClassifier=None
        with open(modelName, "rb") as f:
            NaiveBayesClassifier = pickle.load(f)    

        [predictions,accu]=NaiveBayesClassifier.getAccuracy(x_test,y_test)
        print("Accuracy:",accu)
        

if __name__=="__main__":
    main()


# x_test=tokenizedInput_WithNGrams('Dataset/Stem/imdb_test_text.txt',grams=2)
# y_test=tokenizedInput('Dataset/imdb/imdb_test_labels.txt',tokenize=False)
# y_test=list(map(int,y_test))


# # In[55]:


# NaiveBayesClassifier=NaiveBayes(x=x_train,y=y_train)
# NaiveBayesClassifier.calculateParameters()
# print (len(NaiveBayesClassifier.x_dict))
# print(NaiveBayesClassifier.distinct_x)
# print(len(NaiveBayesClassifier.vocabulary))


# # In[59]:


# NaiveBayesClassifier.purge(threshold=10)
# print(len(NaiveBayesClassifier.x_dict))
# print(NaiveBayesClassifier.distinct_x)
# print(len(NaiveBayesClassifier.vocabulary))


# # In[60]:


# [a,b]=(NaiveBayesClassifier.getAccuracy(x_test,y_test,printConfusionMatrix=False))
# print (b)
# # print(NaiveBayesClassifier.ConfusionMatrix(a,y_test))


# # In[25]:


# NaiveBayesClassifier.ConfusionMatrix(y_test,a)


# # In[251]:


# NaiveBayesClassifier.y_dict


# # In[56]:


# x_train2=tokenizedInput_WithNGrams('Dataset/Stem/imdb_train_text.txt',grams=2)
# y_train2=tokenizedInput('Dataset/imdb/imdb_train_labels.txt',tokenize=False)
# y_train2=list(map(int,y_train2))
# x_test2=tokenizedInput_WithNGrams('Dataset/imdb/imdb_test_text.txt',grams=2)
# y_test2=tokenizedInput('Dataset/imdb/imdb_test_labels.txt',tokenize=False)
# y_test2=list(map(int,y_test2))


# # In[30]:


# NaiveBayesClassifier2=NaiveBayes(x=x_train2,y=y_train2)
# NaiveBayesClassifier2.calculateParameters()
# # print (sum(NaiveBayesClassifier2.wordsInClass))


# # In[32]:


# [a,b]=(NaiveBayesClassifier2.getAccuracy(x_train2,y_train2,printConfusionMatrix=False))
# print (b)


# # In[33]:


# # x_train3=tokenizedInput_WithNGrams('Dataset/Stem/imdb_train_text.txt',grams=3)
# # y_train3=tokenizedInput('Dataset/imdb/imdb_train_labels.txt',tokenize=False)
# # y_train3=list(map(int,y_train3))
# # x_test3=tokenizedInput_WithNGrams('Dataset/imdb/imdb_test_text.txt',grams=3)
# # y_test3=tokenizedInput('Dataset/imdb/imdb_test_labels.txt',tokenize=False)
# # y_test3=list(map(int,y_test3))


# # In[34]:


# # print (len(NaiveBayesClassifier2.x_dict))


# # In[57]:


# x_train1=tokenizedInput_WithNGrams('Dataset/Clean1/imdb_train_text.txt',grams=1)
# y_train1=tokenizedInput('Dataset/imdb/imdb_train_labels.txt',tokenize=False)
# y_train1=list(map(int,y_train1))
# x_test1=tokenizedInput_WithNGrams('Dataset/Clean1/imdb_test_text.txt',grams=1)
# y_test1=tokenizedInput('Dataset/imdb/imdb_test_labels.txt',tokenize=False)
# y_test1=list(map(int,y_test1))


# # In[58]:


# NaiveBayesClassifier1=NaiveBayes(x=x_train1,y=y_train1)
# NaiveBayesClassifier1.calculateParameters()
# print (len(NaiveBayesClassifier1.x_dict))


# # In[59]:


# [a1,b1]=(NaiveBayesClassifier1.getAccuracy(x_train1,y_train1,printConfusionMatrix=False))
# print (b1)


# # In[60]:


# NaiveBayesClassifier1.ConfusionMatrix(y_test1,a1)


# # In[ ]:


# x_train_uni_bi=tokenizedInput_WithNGrams('Dataset/Stem/imdb_train_text.txt',grams=2)
# y_train_uni_bi=tokenizedInput('Dataset/imdb/imdb_train_labels.txt',tokenize=False)
# y_train_uni_bi=list(map(int,y_train_uni_bi))
# x_test_uni_bi=tokenizedInput_WithNGrams('Dataset/Stem/imdb_test_text.txt',grams=2)
# y_test_uni_bi=tokenizedInput('Dataset/imdb/imdb_test_labels.txt',tokenize=False)
# y_test_uni_bi=list(map(int,y_test1))

