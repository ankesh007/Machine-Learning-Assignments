from Regression import Regression
import sys
import numpy as np
import pandas as pd

class LinearRegression(Regression):

	def __init__(self,pd_x,pd_y,normalize_x=True,normalize_y=False):
		super().__init__(pd_x,pd_y,normalize_x,normalize_y)

	def evaluate(self,x):
		return np.matmul(x,self.theta)


def main(path_x,path_y):

	temp_x=pd.read_csv(path_x,header=None,sep=',')
	temp_y=pd.read_csv(path_y,header=None,sep=',')

	myReg=LinearRegression(pd_x=temp_x,pd_y=temp_y,normalize_x=True)
	# print (temp_x.values.shape)
	# print (temp_y.values.shape)
	# print (temp_x.values)
	# print(temp_y.values)


	# print (myReg.x.shape)
	# print (myReg.y.shape)
	myReg.train(log_every_epoch=50,learning_rate=0.001,epsilon=0.00001)
	# print (myReg.train_steps)
	print (myReg.epoch)
	print (myReg.getLoss(myReg.x,myReg.y))
	print (np.concatenate((myReg.predict(temp_x.values),temp_y),axis=1))
	

if __name__=="__main__":

	main(sys.argv[1],sys.argv[2])
