from Regression import Regression
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

class LinearRegression(Regression):

	def __init__(self,pd_x,pd_y,normalize_x=True,normalize_y=False):
		super().__init__(pd_x,pd_y,normalize_x,normalize_y)

	def custom_loss(self,theta0,theta1):
		self.theta=np.array([[theta0],[theta1]])
		return self.getLoss(self.x,self.y)

	# def evaluate(self,x):
	# 	return np.matmul(x,self.theta)
	# Not required because Base Class supports Linear Regression
	
def plotPoints_Hypothesis(myReg,filename):

	plt.figure()
	plt.xlabel("Normalized X")
	plt.ylabel("Y")
	plt.title("Linear Regression- Data and Hypothesis")
	plt.plot(myReg.unappended_x,myReg.y,'ro',myReg.unappended_x,myReg.predict(myReg.dataframe_x.values),'k')
	# plt.plot(myReg.dataframe_x.values,myReg.y,'ro',myReg.dataframe_x.values,myReg.predict(myReg.dataframe_x.values),'k')
	# for unnormalized
	plt.savefig(filename+'.png')

def plotContour(myReg,filename):
	print("**************Contour Analysis*****************")
	myReg.initialise_theta()
	delta = 0.025
	x = np.arange(-0.5, 3.025, delta)
	y = np.arange(-0.5, 3.025, delta)
	X, Y = np.meshgrid(x, y)
	Z=(np.vectorize(myReg.custom_loss))(X,Y)
	plt.figure()
	CS=plt.contour(X,Y,Z)
	plt.clabel(CS, inline=1, fontsize=10)
	plt.title('Contours and Path Traversed')
	plt.xlabel('Theta0')
	plt.ylabel('Theta1')
	myReg.train(log_every_epoch=40,learning_rate=0.0001,epsilon=0.000001,plot=True)
	# plt.ion()
	print("**************Ended-> Click Cross Button*****************")
	plt.show()
	plt.savefig(filename+'.png')

def drawSurface(myReg,filename):
	print("**************Surface Analysis*****************")
	delta = 0.025
	x = np.arange(-5, 5, delta)
	y = np.arange(-5, 5, delta)
	X, Y = np.meshgrid(x, y)
	Z=(np.vectorize(myReg.custom_loss))(X,Y)
	# print (X.shape,Z.shape)

	fig=plt.figure()
	ax = fig.gca(projection='3d')
	surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,linewidth=0.2)

	fig.colorbar(surf, shrink=0.5, aspect=5)
	plt.xlabel("Theta0")
	plt.ylabel("Theta1")
	plt.title("Plotting J(theta) against theta")
	myReg.train(log_every_epoch=40,learning_rate=0.0001,epsilon=0.000001,plot=False,plot3D=True)
	print("**************Ended-> Click Cross Button*****************")
	plt.show()
	plt.savefig(filename+'.png')

def main(path_x,path_y):

	temp_x=pd.read_csv(path_x,header=None,sep=',')
	temp_y=pd.read_csv(path_y,header=None,sep=',')

	myReg=LinearRegression(pd_x=temp_x,pd_y=temp_y,normalize_x=True)

	drawSurface(myReg,"hey")
	plotContour(myReg,"LinearRegressionContours")
	plotPoints_Hypothesis(myReg,"LinearRegressionHypothesis")
	print ("*****Theta******")
	print(myReg.theta)

if __name__=="__main__":

	main(sys.argv[1],sys.argv[2])
