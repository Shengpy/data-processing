import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import random
#---------------------------------------------count number None in data 
df=pd.read_excel('S.xls')
#data=pd.read_csv('data.csv')
#print(df.isnull().sum())

#for col in df.columns:
#    missing_data=df[col].isna().sum()
#    percent_lost_data=int(missing_data/len(df)*100)
#    print('Column {}: stand by: {}% has {}'.format(col,percent_lost_data,missing_data))
#fig,ax=plt.subplots(figsize=(10,8))
#sns.heatmap(df.isna(),cmap='Blues',cbar=False,yticklabels=False)
#plt.show() 

#df.dropna(axis='columns', how='all')               #delete if all row or columne is na 
#df.dropna(axis='rows', thresh=3)                   #y a minimum number of non-null values for the row/column to be kept   
#data.fillna(0)
#data.fillna(method='ffill', axis=1)                # forward-fill to propagate the previous value
#data.fillna(method='bfill')                        #a back-fill to propagate the next values

#from sklearn.impute import SimpleImputer 
#x=df.loc[:,['Password','StudentName','Mobile']].values
#imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
#imputer.fit(x[:,2].reshape(-1, 1))
#x[:,2:3]=imputer.transform(x[:,2].reshape(-1, 1))
#print(x)
#---------------------------------Basic Train
#features=['sqft_lot','yr_built']
#X=data[features]
#drop_list=['price','street','city','statezip','country','date']
#X=data.drop(drop_list,axis=1)
#y=data['price']
#from sklearn.model_selection import train_test_split
#X_train,X_valid,y_train,y_valid=train_test_split(X,y,train_size=0.8,test_size=0.2,random_state=0)

#Way----1---- 
#from sklearn.tree import DecisionTreeRegressor 
#from sklearn.tree import DecisionTreeClassifier 
#dt_model= DecisionTreeRegressor(random_state=1)
#dt_model.fit(X_train,y_train)
#y_preds=dt_model.predict(X_valid.head(1))
#print(pd.DataFrame({'y':y_valid.head(1),'y predict':y_preds}))

#from sklearn.linear_model import LinearRegression
#lr = LinearRegression()
#lr.fit(X_train,y_train)
#y_preds=lr.predict(X_valid.head())
#print(pd.DataFrame({'y':y_valid.head(),'y predict':y_preds}))

#from sklearn.metrics import mean_squared_error 
#mse=mean_squared_error(y_valid.head(),y_preds)
#rmse=np.sqrt(mse)
#print(mse)
#Way----2-----
##from sklearn.ensemble import RandomForestClassifier 
#from sklearn.ensemble import RandomForestRegressor 
#dt_model= RandomForestRegressor(random_state=1)
#dt_model.fit(X_train,y_train)
#y_preds=dt_model.predict(X_valid)
#dt_model.predict([[6969,2021,2341]])       #bên trong là dữ liệu 

#----------------------------------------Mã hóa dữ liệu
#from sklearn.preprocessing import LabelEncoder
#le=LabelEncoder()
#y=['No','Yes','No','No','Yes']
#y=le.fit_transform(y)
#print(y)

#from sklearn.compose import ColumnTransformer
#from sklearn.preprocessing import OneHotEncoder
#ct=ColumnTransformer(transformers=[ ('encoder',OneHotEncoder(),[0])],remainder='passthrough')
#X=ct.fit_transform(X)
#print(X)
#-------------------------------
#sns.pairplot(data,height=2.5)
#plt.tight_layout()

#sns.distplot(data['price'])
#print( data['price'].skew())

#fig,ax=plt.subplots(figsize=(1,1),)
#ax.scatter(x=data['yr_built'],y=data['price'])
#plt.ylabel('price')
#plt.xlabel('yr_built')
#plt.show()

#@@@@@
#from scipy import stats
#from scipy.stats import norm,skew
#sns.distplot(data['price'],fit=norm)
#(mu,sigma)=norm.fit(data['price'])
#print('\n mu={:.2f} and sigma= {:.2f}'.format(mu,sigma))

#plt.legend(['Normal dist.($\mu=$ {:.2f} and $\sigma=$ {:.2f})'.format(mu,sigma)]
#         ,loc='best')
#plt.ylabel('Frequency')
#plt.title('SalePrice distribution')

#fig=plt.figure()
#res=stats.probplot(data['price'],plot=plt)
#plt.show()

#plt.figure(figsize=(10,10))
#cor=data.corr()
#sns.heatmap(cor,annot=True,cmap=plt.cm.PuBu)
#plt.show()

#cor_target=abs(cor['price'])
#relevant_features=cor_target[cor_target>0.2]
#names=[index for index, value in relevant_features.iteritems()]
#names.remove('price')
#print(names)
#print(len(names))

