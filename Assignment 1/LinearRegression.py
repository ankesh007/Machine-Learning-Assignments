from Regression import Regression
import sys
import numpy as np
import pandas as pd

class LinearRegression(Regression):

	def __init__(self,pd_x,pd_y,normalize_x=True,normalize_y=False):
		super().__init__(pd_x,pd_y,normalize_x,normalize_y)

	# def evaluate(self,x):
	# 	return np.matmul(x,self.theta)
	# Not required because Base Class supports Linear Regression


def main(path_x,path_y):

	temp_x=pd.read_csv(path_x,header=None,sep=',')
	temp_y=pd.read_csv(path_y,header=None,sep=',')

	myReg=LinearRegression(pd_x=temp_x,pd_y=temp_y,normalize_x=True)
	myReg.train(log_every_epoch=50,learning_rate=0.01,epsilon=0.0001)
	myReg.solveAnalytically()
	# print (myReg.train_steps)
	# print (myReg.epoch)
	# print (myReg.getLoss(myReg.x,myReg.y))
	
	# temp=np.concatenate((myReg.predict(temp_x.values),temp_y.values),axis=1)
	# temp=np.concatenate((temp,myReg.predict(temp_x.values)),axis=1)


if __name__=="__main__":

	main(sys.argv[1],sys.argv[2])
