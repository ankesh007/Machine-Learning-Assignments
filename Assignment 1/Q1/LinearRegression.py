from Regression import Regression
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

epsilon=0.0000001
learning_rate=0.001
log_every_epoch=1

class LinearRegression(Regression):

	def __init__(self,pd_x,pd_y,normalize_x=True,normalize_y=False):
		super().__init__(pd_x,pd_y,normalize_x,normalize_y)

	def custom_loss(self,theta0,theta1):
		self.theta=np.array([[theta0],[theta1]])
		return self.getLoss(self.x,self.y)

def plotPoints_Hypothesis(myReg,filename):

	myReg.initialise_theta()
	plt.figure()
	plt.xlabel("Normalized X")
	plt.ylabel("Y")
	plt.title("Linear Regression- Data and Hypothesis")
	myReg.train(log_every_epoch=log_every_epoch,learning_rate=learning_rate,epsilon=epsilon)
	plt.plot(myReg.unappended_x,myReg.y,'ro',myReg.unappended_x,myReg.predict(myReg.dataframe_x.values),'k')
	plt.savefig(filename+'.png')

def drawHypothesisAnalytic(myReg,filename):

	myReg.initialise_theta()
	plt.figure()
	plt.xlabel("Normalized X")
	plt.ylabel("Y")
	plt.title("Linear Regression- Data and Hypothesis")
	myReg.solveAnalytically()
	plt.plot(myReg.unappended_x,myReg.y,'ro',myReg.unappended_x,myReg.predict(myReg.dataframe_x.values),'k')
	plt.savefig(filename+'.png')

def plotContour(myReg,filename):
	print("**************Contour Analysis*****************")
	delta = 0.025
	x = np.arange(-0.5, 1.5, delta)
	y = np.arange(-0.5, 1.5, delta)
	X, Y = np.meshgrid(x, y)
	Z=(np.vectorize(myReg.custom_loss))(X,Y)
	plt.figure()
	CS=plt.contour(X,Y,Z)
	plt.clabel(CS, inline=1, fontsize=10)
	plt.title('Contours and Path Traversed')
	plt.xlabel('Theta0')
	plt.ylabel('Theta1')
	myReg.initialise_theta()
	myReg.train(log_every_epoch=log_every_epoch,learning_rate=learning_rate,epsilon=epsilon,plot=True)
	plt.savefig(filename+'.png')
	print("**************Ended-> Click Cross Button*****************")
	plt.show()

def drawSurface(myReg,filename):
	print("**************Surface Analysis*****************")
	delta = 0.025
	x = np.arange(-1.5, 3, delta)
	y = np.arange(-1.5, 3, delta)
	X, Y = np.meshgrid(x, y)
	Z=(np.vectorize(myReg.custom_loss))(X,Y)
	fig=plt.figure()
	ax = fig.gca(projection='3d')
	surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,linewidth=0.2)
	fig.colorbar(surf, shrink=0.5, aspect=5)
	plt.xlabel("Theta0")
	plt.ylabel("Theta1")
	plt.title("Plotting J(theta) against theta")
	myReg.initialise_theta()
	myReg.train(log_every_epoch=log_every_epoch,learning_rate=learning_rate,epsilon=epsilon,plot=False,plot3D=True)
	plt.savefig(filename+'.png')
	print("**************Ended-> Click Cross Button*****************")
	plt.show()

def main(path_x,path_y,draw_surface,draw_contour,draw_hypothesis,solve_analytically):

	temp_x=pd.read_csv(path_x,header=None,sep=',')
	temp_y=pd.read_csv(path_y,header=None,sep=',')

	myReg=LinearRegression(pd_x=temp_x,pd_y=temp_y)

	if(draw_surface=='True'):
		drawSurface(myReg,"Surface:Learning Rate="+str(learning_rate))

	if(draw_contour=='True'):
		plotContour(myReg,"Contours:Learning Rate="+str(learning_rate))

	if(draw_hypothesis=='True'):
		plotPoints_Hypothesis(myReg,"Hypothesis:Learning Rate="+str(learning_rate))

	if(solve_analytically=='True'):
		drawHypothesisAnalytic(myReg,"AnalyticHypothesis")
	
	print ("*****Theta******")
	print(myReg.theta)

if __name__=="__main__":

	if(len(sys.argv)<8):
		print("Usage: <script> <path_x> <path_y> <draw_surface> <draw_contour> <draw_hypothesis> <solve_analytically> <learning_rate>")
		exit()

	learning_rate=float(sys.argv[7])
	# print(learning_rate)
	main(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5],sys.argv[6])