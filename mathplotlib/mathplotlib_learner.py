import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd 
df=pd.read_excel('S.xls')
#x=np.arange(1,5)
#y=x**2
#plt.figure(figsize=(15,5))                 #changesize 
#plt.subplot(1,2,1) #x,y,position
#'go','r^','bD--','b*','y*'
#plt.plot([1,2,3,6,10],[1,4,7,8,8],"go",label='greenDot') 
#plt.legend(loc='best')                      # add note 
#plt.title("Sheng")
#plt.subplot(1,2,2)
#plt.plot(x,y,'r^')
#plt.suptitle("Super Sheng")
##plt.xlabel("x label")
##plt.ylabel("y label")
#plt.show() 

#fix,ax=plt.subplots(nrows=2,ncols=2,figsize=(6,6))
#ax[0,1].plot([1,2,3,6,10],[1,4,7,8,8],"go")
#ax[1,0].plot(x,y,'r^',color="red")
#ax[0,0].plot([1,2,3,6,10],[1,4,7,8,8],color='blue',linestyle='dashed')
#ax[0,1].set(title="Sheng2",xlabel='x',ylabel='y')
#ax[1,0].set(title="Sheng3",xlabel='x',ylabel='y')
#plt.suptitle("Super Sheng")
#plt.show() 

#x=np.linspace(0,10,1000)
#plt.plot(x,np.sin(x),color="green",linestyle='dashed')
#plt.plot(x,np.cos(x),color="blue")
#-------------------------------------------------------------3D-----------------------
#height=np.array([1,2,3,6,10])
#weight=np.array([1,4,7,8,8])
#ax=plt.axes(projection='3d')
#ax.scatter3D(height,weight)
#ax.set_xlabel("height")
#ax.set_ylabel("weight")
#-------------------------------------------------------------bar---------------
#plt.bar(["Sheng","Anh"],[50,50],color="red")

#soft_drink_prices={"Coke":10,'Pessi':12,'Sprite':8}
#fig,ax=plt.subplots()
#ax.barh(list(soft_drink_prices.keys()),soft_drink_prices.values())
#ax.bar(soft_drink_prices.keys(),soft_drink_prices.values())
#ax.set(title='bach hoa xanh',ylabel='Price')
#------------------------------------------------------------thống kê 
#sns.set()
#plt.hist(df['Password'],bins=30)  #bin number collum show 
#sns.displot(df['Password'])
#------------------------------------------------------------Biểu đồ tròn
#len_female=df['Nữ'].isna().sum()
#len_male=len(df['Nữ'])-len_female
#labels=['Male','female']
#arr=np.array([len_male,len_female])
#h=df.fillna(0)
#plt.pie(arr,labels=labels,shadow=False )
#-----------------------------------------------------------------Example
#x = np.linspace(0, 5, 50)
#y = np.linspace(0, 5, 50)[:, np.newaxis]
#z = np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)
#print(z)
#plt.imshow(z, origin='lower', extent=[0, 5, 0, 5],cmap='viridis')
#plt.colorbar()
#plt.show()

#mean = [0, 0]
#cov = [[1, 2],[2, 5]]
#X = np.random.multivariate_normal(mean, cov, 100)
#indices = np.random.choice(X.shape[0], 20, replace=False)
#selection = X[indices]
#sns.set() # for plot styling
##plt.plot(X[:, 0], X[:, 1],'go')
#plt.scatter(X[:, 0], X[:, 1], alpha=0.3)
#plt.scatter(selection[:, 0], selection[:, 1],facecolor='red', s=200,alpha=0.1)
#plt.show()
#-------------------------------- mật độ phân bố
#np.random.seed(42)
#x = np.random.randn(100)
#bins = np.linspace(-5, 5, 20)

#counts = np.zeros_like(bins)                            #faster 
#i = np.searchsorted(bins, x) #return index of values similar to x 
#np.add.at(counts, i, 1)
#plt.plot(bins, counts, linestyle='dashdot')

#counts, edges = np.histogram(x, bins)                    #slower 

#plt.hist(x, bins, histtype='step')                       #the same 
#plt.show()
#----------------------------------

