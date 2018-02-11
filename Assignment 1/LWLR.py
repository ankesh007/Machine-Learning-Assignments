from Regression import Regression
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class LocallyWeightedLinearRegression(Regression):

	def __init__(self,pd_x,pd_y,normalize_x=True,normalize_y=False):
		super().__init__(pd_x,pd_y,normalize_x,normalize_y)
		# self.tau=tau
		# self.weight=None

	def solveAnalytically(self,weight=None):

		if(weight is None):
			print ("Error:Provide Weight Matrix(weight=)")
			return

		transpose_x=np.transpose(self.x)
		temp_var=np.matmul(transpose_x,weight)
		temp_lef=np.matmul(temp_var,self.x)
		temp_rig=np.matmul(temp_var,self.y)
		temp_lef_inv=np.linalg.inv(temp_lef)
		self.theta=np.matmul(temp_lef_inv,temp_rig)
		# print (self.theta)

	def predict(self,x,tau):

		[instances,parameters]=self.x.shape

		temp_x=np.copy(x)
		if(self.has_normalized_x==True):
			temp_x=self.normalize(temp_x,self.x_mu,self.x_sigma)

		modified_x=self.append_1(temp_x)
		diff_x=self.x-modified_x
		norm_x=self.L2normMatrixRowWise(diff_x)
		norm_x=np.square(norm_x)
		norm_x=norm_x/(2*tau*tau)
		weight=np.exp(-1*norm_x)
		# print(weight)
		weight=np.diag(weight[:,0])
		# print(weight.shape)

		self.solveAnalytically(weight=weight)
		temp_eval=self.evaluate(modified_x)
		if(self.has_normalized_y==True):
			temp_eval=self.unnormalize(temp_eval,self.y_mu,self.y_sigma)

		return temp_eval

def plotPoints_Hypothesis(myReg,prediction,filename):

	plt.figure(1)
	fig, ax = plt.subplots()
	plt.xlabel("Normalized X")
	plt.ylabel("Normalized Y")
	plt.title("LWLR- Data and Hypothesis")
	# print(myReg.unappended_x[(myReg.y[:,0]==1)])
	argsort_val=myReg.unappended_x[:,0].argsort()
	ax.plot(myReg.unappended_x,myReg.y,'ro',label='data')
	# print(myReg.unappended_x[argsort_val].shape)
	# print(argsort_val)
	# print(myReg.unappended_x[argsort_val][:,:,0])
	ax.plot(myReg.unappended_x[argsort_val],prediction[argsort_val],'k',label='hypothesis')

	legend = ax.legend(loc='upper left',fontsize='x-small')
	legend.get_frame().set_facecolor('#00FFCC')
	plt.savefig(filename+'.png')


def main(path_x,path_y):

	temp_x=pd.read_csv(path_x,header=None,sep=',')
	temp_y=pd.read_csv(path_y,header=None,sep=',')

	myReg=LocallyWeightedLinearRegression(pd_x=temp_x,pd_y=temp_y)

	temp_x_val=temp_x.values
	[instances,_]=temp_x_val.shape
	print (instances)
	prediction=np.zeros([0,1])

	for i in range(instances):
		query_x=myReg.predict(temp_x_val[i:i+1],100)
		prediction=np.concatenate((prediction,query_x),axis=0)
		# print(query_x)

	plotPoints_Hypothesis(myReg,prediction,'LWLR')
	# print(np.concatenate((prediction,temp_y.values),axis=1))
	# print(myReg.predict(x=temp_x.values[2:3,0:1],tau=0.2),myReg.y[2])
	# myReg.solveAnalytically()
	# myReg.train(log_every_epoch=50,learning_rate=0.01,epsilon=0.0001)
	# print (myReg.train_steps)
	# print (myReg.epoch)
	# print (myReg.getLoss(myReg.x,myReg.y))
	
	# temp=np.concatenate((myReg.predict(temp_x.values),temp_y.values),axis=1)
	# temp=np.concatenate((temp,myReg.predict(temp_x.values)),axis=1)


if __name__=="__main__":

	main(sys.argv[1],sys.argv[2])
