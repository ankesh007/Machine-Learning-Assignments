import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# Alaska is numbered 1
class GDA:

	def __init__(self,pd_x,pd_y,map_to_1='Alaska',map_to_0='Canada',normalize_x=True,normalize_y=False):
		self.dataframe_x=pd_x
		self.dataframe_y=pd_y
		temp_x=self.dataframe_x.values
		temp_y=self.dataframe_y.values

		if(normalize_x):
			self.x_mu=self.getMean(temp_x)
			self.x_sigma=self.getStd(temp_x)
			temp_x=self.normalize(temp_x,self.x_mu,self.x_sigma)
		else:
			self.x_mu=0
			self.x_sigma=1.0

		self.unappended_x=np.copy(temp_x)
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

	def getMean(self,x):
		return np.mean(x,axis=0)

	def getStd(self,x):
		return np.std(x,axis=0)

	def normalize(self,x,mu,sigma):
		return (x-mu)/sigma

	def getMu(self,for_y_equal):
		indicator=(self.y==for_y_equal)
		indicator_x=self.x*indicator
		summed_x=np.sum(indicator_x,axis=0)
		sum_indicator=np.sum(indicator)
		return (summed_x/sum_indicator)

	def getPhi(self):
		indicator=(self.y==1)
		[instances,_]=self.y.shape
		sum_indicator=np.sum(indicator).astype(float)
		return (sum_indicator/instances)

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
		temp=(np.matmul(transpose_x,self.x)/sum_indicator)
		temp=np.linalg.inv(temp)
		return temp

	def getCommonSigma(self):
		indicator=(self.y==0)
		temp_x=np.copy(self.x)

		temp_x-=(indicator*self.mu0+(1-indicator)*self.mu1)

		transpose_x=np.transpose(temp_x)
		sum_indicator=(self.x.shape)[0]
		temp=(np.matmul(transpose_x,temp_x)/sum_indicator)
		temp=np.linalg.inv(temp)
		return temp

	def setParameters(self,common_sigma=False):
		self.phi=self.getPhi()
		self.mu1=self.getMu(for_y_equal=1)
		self.mu0=self.getMu(for_y_equal=0)
		if(common_sigma==False):
			self.sigma0=self.getSigma(0)
			self.sigma1=self.getSigma(1)

		else:
			self.sigma=self.getCommonSigma()

	def getX2Linear(self,X1):
		mu1=self.mu1
		mu0=self.mu0
		sigma=self.sigma
		phi=self.phi
		x2_coeff=(mu1[1]-mu0[1])*sigma[1,1]
		c1=math.log((1-phi)/(phi))
		c2=0.5*(mu1[0]*sigma[0,0]*mu1[0]+mu1[1]*sigma[1,1]*mu1[1]-mu0[0]*sigma[0,0]*mu0[0]-mu0[1]*sigma[1,1]*mu0[1])
		linear=(mu0[0]*sigma[0,0]-mu1[0]*sigma[0,0])*X1
		calc_x2=(c1+c2+linear)/x2_coeff
		return calc_x2

	def getTuple(self,x,mu,sigma):
		c2=(x-mu[0])*(x-mu[0])*sigma[0,0]
		k1=(x-mu[0])*sigma[0,1]
		k2=(x-mu[0])*sigma[1,0]
		coeff_x2=sigma[1,1]
		coeff_x=-2*mu[1]*sigma[1,1]+k1+k2
		coeff_c=c2+mu[1]*mu[1]*sigma[1,1]-mu[1]*(k1+k2)
		return (coeff_x2,coeff_x,coeff_c)


	def getX2quad(self,X1):
		# print(X1.shape)
		mu1=self.mu1
		mu0=self.mu0
		sigma0=self.sigma0
		sigma1=self.sigma1
		phi=self.phi

		(a1,b1,c1)=self.getTuple(X1,mu1,sigma1)
		(a2,b2,c2)=self.getTuple(X1,mu0,sigma0)

		def getDet(sigma):
			return abs(sigma[0,0]*sigma[1,1]-sigma[0,1]*sigma[1,0])

		detsigma1=getDet(sigma1)
		detsigma0=getDet(sigma0)
		cc=math.log(phi/(1-phi))
		cc=cc-0.5*math.log(detsigma1/detsigma0)

		a=a1-a2
		b=b1-b2
		c=c1-c2-cc

		x2=(-b+math.sqrt(b*b-4*a*c))/(2*a)
		x2_=(-b-math.sqrt(b*b-4*a*c))/(2*a)

		return (x2,x2_)

def plotPoints_LinearHypothesis(myReg,filename):

	plt.figure(1)
	fig, ax = plt.subplots()
	plt.xlabel("Normalized X1")
	plt.ylabel("Normalized X2")
	plt.title("GDA- Data and Linear Hypothesis")
	ax.plot(myReg.unappended_x[(myReg.y[:,0]==1)][:,np.array([True,False])],myReg.unappended_x[(myReg.y[:,0]==1)][:,np.array([False,True])],'rx',label=myReg.map_to_1)
	ax.plot(myReg.unappended_x[(myReg.y[:,0]==0)][:,np.array([True,False])],myReg.unappended_x[(myReg.y[:,0]==0)][:,np.array([False,True])],'bo',label=myReg.map_to_0)
	min_x=np.amin(myReg.unappended_x,axis=0)
	max_x=np.amax(myReg.unappended_x,axis=0)
	arr=np.arange(min_x[0],max_x[0],0.01)
	ax.plot(arr,myReg.getX2Linear(arr),'k',label='Hypothesis')

	legend = ax.legend(loc='upper left',fontsize='x-small')
	legend.get_frame().set_facecolor('#00FFCC')
	plt.savefig(filename+'.png')


def plotPoints_QuadHypothesis(myReg,filename,other_quad_part,linear=True):

	plt.figure(1)
	fig, ax = plt.subplots()
	plt.xlabel("Normalized X1")
	plt.ylabel("Normalized X2")
	plt.title("GDA- Data and Quadratic Hypothesis")
	ax.plot(myReg.unappended_x[(myReg.y[:,0]==1)][:,np.array([True,False])],myReg.unappended_x[(myReg.y[:,0]==1)][:,np.array([False,True])],'rx',label=myReg.map_to_1)
	ax.plot(myReg.unappended_x[(myReg.y[:,0]==0)][:,np.array([True,False])],myReg.unappended_x[(myReg.y[:,0]==0)][:,np.array([False,True])],'bo',label=myReg.map_to_0)
	min_x=np.amin(myReg.unappended_x,axis=0)
	max_x=np.amax(myReg.unappended_x,axis=0)
	arr=np.arange(min_x[0],max_x[0],0.01)

	[c,d]=((np.vectorize(myReg.getX2quad))(arr))
	ax.plot(arr,d,'k',label='Hypothesis')
	if(other_quad_part==True):
		ax.plot(arr,c,'k')

	if(linear==True):
		ax.plot(arr,myReg.getX2Linear(arr),'y',label='Hypothesis Linear')

	legend = ax.legend(loc='upper left',fontsize='x-small')
	legend.get_frame().set_facecolor('#00FFCC')
	plt.savefig(filename+'.png')


def main(path_x,path_y,draw_other_quadratic=False):

	temp_x=pd.read_csv(path_x,header=None,sep=r" *",engine='python')
	temp_y=pd.read_csv(path_y,header=None,sep=r" *",engine='python')

	myReg=GDA(pd_x=temp_x,pd_y=temp_y,normalize_x=True)
	myReg.setParameters(common_sigma=True)
	myReg.setParameters(common_sigma=False)
	print(myReg.sigma0)
	print(myReg.sigma1)
	print(myReg.mu1)
	print(myReg.mu0)
	print(myReg.sigma)
	print(myReg.phi)

	# exit()	
	plotPoints_QuadHypothesis(myReg,"GDAQuad",draw_other_quadratic)
	plotPoints_LinearHypothesis(myReg,"GDALinear")

if __name__=="__main__":

	if(len(sys.argv)<3):
		print("Usage: <script> <x_data_path> <y_data_path> <Draw_Other_quadratic_part>")
		exit()

	if(len(sys.argv)==4):
		if(sys.argv[3]=='False'):
			main(sys.argv[1],sys.argv[2],False)
		else:
			main(sys.argv[1],sys.argv[2],True)

	else:
		main(sys.argv[1],sys.argv[2])
