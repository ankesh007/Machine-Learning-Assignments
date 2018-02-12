from Regression import Regression
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

epsilon=0.000001

class LogisticRegression(Regression):

	def __init__(self,pd_x,pd_y,normalize_x=True,normalize_y=False):
		super().__init__(pd_x,pd_y,normalize_x,normalize_y)

	def eval_x2(self,x):
		temp_x2=self.theta[0]+self.theta[1]*x
		temp_x2/=self.theta[2]
		temp_x2=-1*temp_x2
		return temp_x2

	def evaluate(self,x):
		evaluated=np.matmul(x,self.theta)
		evaluated=np.exp(-1*evaluated)
		evaluated=1/(1+evaluated)
		return evaluated

	def getGradient(self,x,y):
		aux_loss=(y-self.evaluate(x))
		aux_grad=np.matmul(np.transpose(x),aux_loss)
		return (aux_grad)

	def getHessian(self):
		x_transpose=np.transpose(self.x)
		return (-1*np.matmul(x_transpose,self.x))

	def trainNewton(self,epsilon=0.1):

		prev_theta=np.copy(self.theta)
		Trained=False

		while(Trained==False):

			self.epoch=self.epoch+1
			gradient=self.getGradient(self.x,self.y)
			hessian=self.getHessian()
			hessian_inv=np.linalg.inv(hessian)
			hess_inv_grad_prod=np.matmul(hessian_inv,gradient)
			self.theta=self.theta-hess_inv_grad_prod
			cur_theta=np.copy(self.theta)
			if(np.linalg.norm(prev_theta-cur_theta)<epsilon):
				Trained=True
			prev_theta=cur_theta



def plotPoints_Hypothesis(myReg,filename):

	plt.figure(1)
	fig, ax = plt.subplots()
	plt.xlabel("Normalized X1")
	plt.ylabel("Normalized X2")
	plt.title("Logistic Regression- Data and Hypothesis")
	# print(myReg.unappended_x[(myReg.y[:,0]==1)])
	ax.plot(myReg.unappended_x[(myReg.y[:,0]==1)][:,np.array([True,False])],myReg.unappended_x[(myReg.y[:,0]==1)][:,np.array([False,True])],'rx',label='1')
	ax.plot(myReg.unappended_x[(myReg.y[:,0]==0)][:,np.array([True,False])],myReg.unappended_x[(myReg.y[:,0]==0)][:,np.array([False,True])],'bo',label='0')
	min_x=np.amin(myReg.unappended_x,axis=0)
	max_x=np.amax(myReg.unappended_x,axis=0)
	arr=np.arange(min_x[0],max_x[0],0.01)
	ax.plot(arr,myReg.eval_x2(arr),'k',label='Hypothesis')

	legend = ax.legend(loc='upper center',fontsize='x-small')
	legend.get_frame().set_facecolor('#00FFCC')
	plt.savefig(filename+'.png')


def main(path_x,path_y):

	temp_x=pd.read_csv(path_x,header=None,sep=',')
	temp_y=pd.read_csv(path_y,header=None,sep=',')

	myReg=LogisticRegression(pd_x=temp_x,pd_y=temp_y,normalize_x=True)
	myReg.trainNewton(epsilon=epsilon)
	print("*****Theta******")
	print(myReg.theta)
	print("*****Epochs*****")
	print(myReg.epoch)
	plotPoints_Hypothesis(myReg,"LogisticRegression-Err="+str(epsilon))

if __name__=="__main__":

	if(len(sys.argv)<3):
		print("Usage: <script> <x_data_path> <y_data_path> <epsilon-optional>")
		exit()

	if(len(sys.argv)==4):
		epsilon=float(sys.argv[3])

	main(sys.argv[1],sys.argv[2])
