
import numpy as np
import pandas as pd
import sys
import pickle


class SVM:
    
    def __init__(self,x,y,normalize_x=True):
        self.x=np.copy(x).astype(float)
        self.y=np.copy(y)
        [instances,dimensions]=x.shape
        self.dimensions=dimensions
        self.instances=instances
        self.normalize_param=np.ones((1,dimensions))
        self.normalize_x=normalize_x
        label=np.unique(y).shape[0]
        self.label=label
        self.weight=np.zeros((self.label,self.label,dimensions))
        self.bias=np.zeros((self.label,self.label))
        self.epoch=np.zeros((self.label,self.label))
        self.train_steps=np.zeros((self.label,self.label))
        self.isTrained=np.zeros((self.label,self.label))

        if(normalize_x==True):
            self.normalize_param*=255
            self.x=self.normalize(self.x)
        
    def normalize(self,x):
        return (x/self.normalize_param)
    
    def resetParam(self,i,j):
        self.weight[i,j,:]*=0
        self.bias[i,j]=0
        self.epoch[i,j]=0
        self.train_steps[i,j]=0
        self.isTrained[i,j]=0
        return
    
    def evaluate(self,x,y,weight,bias):
        return (np.matmul(x,weight)+bias)*y
    
    def getLoss(self,x,y,lamda,C,weight,bias):
        loss=np.matmul(np.transpose(weight),weight)*lamda
        evaluate=1-self.evaluate(x=x,y=y,weight=weight,bias=bias)
        mask=(evaluate>0).astype(int)
        evaluate*=mask
        loss+=C*np.sum(evaluate)
        return loss
    
    def getWeight(self,i,j):
        return self.weight[i,j,:][:,np.newaxis]
    
    def train1Classier(self,weight,bias,train_steps,epoch,x=None,y=None,lamda=1,C=1,max_steps=4000,batch_mode=True,batch_size=100,epsilon=0.001,log_every=1000):

        if(x is None):
            x=self.x
        if(y is None):
            y=self.y
            
        xy=-(x*y)
        [instances,dim]=x.shape
        [instances,ydim]=y.shape
        
        if(batch_mode==False):
            batch_size=instances
        
        prev_loss=self.getLoss(x=x,y=y,lamda=lamda,C=C,weight=weight,bias=bias)
            
        Trained=False    
        cur_steps=0
        while(Trained==False and cur_steps<max_steps):
            cur_instance=0
            epoch+=1
            
            while(cur_instance<instances and cur_steps<max_steps):
                train_steps+=1
                cur_steps+=1
                eta=1.0/(cur_steps)
                up_lim=min(instances,cur_instance+batch_size)
                batch_x=x[cur_instance:up_lim,0:dim]
                batch_y=y[cur_instance:up_lim,0:ydim]
                batch_xy=xy[cur_instance:up_lim,0:dim]
                examples=up_lim-cur_instance
                cur_instance=up_lim
                evaluate=self.evaluate(batch_x,batch_y,weight,bias)
                mask=((1-evaluate)>=0).astype(int)
                weight=(1-eta*lamda)*weight - (eta*C*(np.sum(batch_xy*mask,axis=0)[:,np.newaxis]))
                bias=bias+((np.sum(mask*batch_y))*C*eta)
                
                if(train_steps%log_every==0):
                    print(train_steps)
#                     print("Batch Loss at Steps=",train_steps," is ",self.getLoss(x,y,lamda,C,weight,bias))
            
            cur_loss=self.getLoss(x,y,lamda,C,weight,bias)
            if(abs(cur_loss-prev_loss)<epsilon):
                prev_loss=cur_loss
                Trained=True
                
            prev_loss=cur_loss
        
        print("Loss on Ending:",prev_loss)

        return (weight,bias,epoch,train_steps,Trained)
    
    def trainIJclassifier(self,i,j,reset=False):
        
        if(reset==True):
            self.resetParam(i,j)
        
        print("********************************************************")
        print(i," ",j," ","Classifier")
        
        if(self.isTrained[i,j]==1):
            return
        
        aux_arr=(self.y==i).astype(int)+(self.y==j).astype(int)

        newx=self.x[aux_arr[:,0]>0]
        newy=self.y[aux_arr[:,0]>0]
        newy2=np.copy(newy)
        newy=((newy2==i).astype(int))-((newy2==j).astype(int))

        (weight,bias,epoch,train_steps,Trained)=self.train1Classier(weight=self.getWeight(i,j),
                                          bias=self.bias[i,j],
                                          x=newx,
                                          y=newy,
                                          train_steps=self.train_steps[i,j],
                                          epoch=self.epoch[i,j])
        
        self.weight[i,j,:]=weight[:,0]
        self.bias[i,j]=bias
        self.epoch[i,j]=epoch
        self.train_steps[i,j]=train_steps
        
        if(Trained==True):
            self.isTrained[i,j]=1
        
    def train(self):
        for i in range(self.label):
            for j in range(i+1,self.label,1):
                self.trainIJclassifier(i,j)

    def getClass(self,x,i,j):
        weight=self.getWeight(i,j)
        
        if(np.matmul(weight.T,x)+self.bias[i,j]>0):
            return 1
        else:
            return 0
                
    def predictInstance(self,x):
        
        count=[]
        for i in range(self.label):
            count.append(0)
            
        for i in range(self.label):
            for j in range(i+1,self.label,1):
                aux_class=self.getClass(x,i,j)
                if(aux_class==1):
                    count[i]+=1
                else:
                    count[j]+=1
        
        coun=-1
        lab=-1
        
        for i in range(self.label):

            if(coun<=count[i]):
                coun=count[i]
                lab=i
        return lab
    
    def predict(self,x):
        predictions=[]
        
        x=(x.astype(float))/self.normalize_param
        instances=x.shape[0]
        
        for i in range(instances):
            predictions.append(self.predictInstance(x[i][::,np.newaxis]))
        
        return np.asarray(predictions)[:,np.newaxis]
            
    def getAccuracy(self,x,y):
        predictions=self.predict(x)
        
        mask_correct=np.sum((predictions==y).astype(int))
        instances=y.shape[0]
        
        return float(mask_correct)/instances
    
    def saveModel(self,filename):
        file_handler=open(filename,"w")
        pickle.dump(self,file_handler)
        f.close()
        

# In[566]:

def parseDataframe(dataframe):
    values=dataframe.values
    [instances,dim]=values.shape
    x=values[:,0:dim-1]
    y=values[:,dim-1:dim]
    return (x,y)


# In[567]:

def main():


    if(len(sys.argv)<4):
        print ("Usage: <script> <mode> <inputfilename> <modelName> <outputFileName>")
        print("<outputFileName> is required for mode=1")
        print ("mode -> 0 = Train | 1 = Predict | 2 = Accuracy")
        print ("Mode 0 and 2 expects labels to be appended as last column in the file")
        exit()

    mode=int(sys.argv[1])
    modelName=sys.argv[3]

    if(mode==0):

        trainpd=pd.read_csv(sys.argv[2],header=None,sep=',')
        (x_train,y_train)=parseDataframe(trainpd)
        SVM_instance=SVM(x_train,y_train)
        SVM_instance.train()
        f=open(modelName,"wb")
        pickle.dump(SVM_instance,f)
        f.close()
        
    elif(mode==1):
        if(len(sys.argv)<5):
            print ("Wrong format for mode=1. Exitting")
            exit()
        outputFileName=sys.argv[4]
        predictpd=pd.read_csv(sys.argv[2],header=None,sep=',')
        predictpd=predictpd.values
        SVM_instance=None
        with open(modelName, "rb") as f:
            SVM_instance = pickle.load(f)    

        predictions=SVM_instance.predict(predictpd)
        np.savetxt(outputFileName,predictions,fmt="%d")

    else:
        trainpd=pd.read_csv(sys.argv[2],header=None,sep=',')
        (x_train,y_train)=parseDataframe(trainpd)
        SVM_instance=None
        with open(modelName, "rb") as f:
            SVM_instance = pickle.load(f)    

        print("Accuracy:",SVM_instance.getAccuracy(x_train,y_train))
        

if __name__=="__main__":
    main()

