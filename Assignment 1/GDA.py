import sys
import numpy as np
import pandas as pd

# Alaska is numbered 1

class GDA:

	def __init__(self,pd_x,pd_y,map_to_1='Alaska',map_to_0='Canada',normalize_x=True,normalize_y=False):
		self.dataframe_x=pd_x
		self.dataframe_y=pd_y
		temp_x=self.dataframe_x.values
		temp_y=self.dataframe_y.values
		self.x=temp_x.astype(np.float32)
		self.y=(temp_y==map_to_1).astype(np.float32)
		self.map_to_1=map_to_1
		self.map_to_0=map_to_0
		self.phi=None
		self.mu1=None
		self.mu0=None
		self.sigma1=None
		self.sigma0=None
		self.sigma=None

	def getMean(self,for_y_equal):
		indicator=(self.y==for_y_equal)
		indicator_x=self.x*indicator
		summed_x=np.sum(indicator_x,axis=0)
		sum_indicator=np.sum(indicator)
		return (summed_x/sum_indicator)

	def getPhi(self):
		indicator=(self.y==1)
		[instances,_]=self.y.shape
		sum_indicator=np.sum(indicator).astype(float)
		return (indicator/instances)

	def getSigma(self,for_y_equal):
		indicator=(self.y==for_y_equal)
		temp_x=np.copy(self.x)
		
		if(for_y_equal==0):
			temp_x-=self.mu0
		else:
			temp_x-=self.mu1

		temp_x*=indicator

		transpose_x=np.transpose(temp_x)
		sum_indicator=np.sum(indicator)
		return (np.matmul(transpose_x,self.x))/sum_indicator

	def getCommonSigma(self):
		indicator=(self.y==0)
		temp_x=np.copy(self.x)

		temp_x-=(indicator*self.mu0+(1-indicator)*self.mu1)

		transpose_x=np.transpose(temp_x)
		sum_indicator=(self.x.shape)[0]
		return (np.matmul(transpose_x,self.x))/sum_indicator

	def setParameters(self,common_sigma=False):
		self.phi=self.getPhi()
		self.mu1=self.getMean(for_y_equal=1)
		self.mu0=self.getMean(for_y_equal=0)
		if(common_sigma==False):
			self.sigma0=self.getSigma(0)
			self.sigma1=self.getSigma(1)

		else:
			self.sigma=self.getCommonSigma()

def main(path_x,path_y):

	temp_x=pd.read_csv(path_x,header=None,sep='  ')
	temp_y=pd.read_csv(path_y,header=None,sep='  ')

	myReg=GDA(pd_x=temp_x,pd_y=temp_y,normalize_x=True)
	myReg.setParameters(common_sigma=True)
	myReg.setParameters(common_sigma=False)


	# print(myReg.x.shape)
	# print(myReg.x.dtype)
	# print(myReg.mu0)
	# print(myReg.mu1)
	# print(myReg.sigma)
	# print(myReg.sigma0)
	# print(myReg.sigma1)
	# print(myReg.x.astype(np.float32))
	# print(myReg.y)




	# print (myReg.x)
	# print (myReg.y)

	# myReg.trainNewton(epsilon=0.000001)
	# print(myReg.theta)
	# print(myReg.epoch)
	# myReg.solveAnalytically()
	# print (myReg.train_steps)
	# print (myReg.epoch)
	# print (myReg.getLoss(myReg.x,myReg.y))
	
	# temp=np.concatenate((myReg.predict(temp_x.values),temp_y.values),axis=1)
	# temp=np.concatenate((temp,myReg.predict(temp_x.values)),axis=1)
	# print(temp)


if __name__=="__main__":

	main(sys.argv[1],sys.argv[2])
