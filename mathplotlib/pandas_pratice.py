import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import random
df=pd.read_excel('S.xls')

#--------------------------------------------đếm học sinh trùng tên
#df['number_student']=1
#t=df.groupby('StudentName')['number_student'].sum().sort_values()
#print(t.loc[t.values>1])
#--------------------------------------------tỉ lệ giới tính 
#df['Gender']=df.Nữ
#df.pop('Nữ')
#df.loc[df.Gender.isnull(),'Gender']='Male'
#df.loc[df.Gender=='x','Gender']='Female'
#print(df.pivot_table( index='Lớp', columns='Gender', aggfunc={'Gender':'count'}))
#sns.set() 
#df.pivot_table( index='Lớp', columns='Gender', aggfunc={'Gender':'count'}).plot() 
#plt.ylabel('rate Gender in every class')
#plt.show()
print(df.corr())