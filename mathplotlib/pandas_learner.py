import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import random
df=pd.read_excel('S.xls')
#train=df.pop('Username')

#print(df.loc[0],train.loc[0])
#print(df.loc[( df.Nữ=='x')&(df.Lớp=='11A4'),['Password','StudentName']])       # |  &
#print(df.iloc[:,:-2].values  )           #df.iloc[[9]]   #df.iloc[1:9:-1]   #phần tử thứ n

#print(df.describe() )          #describe(include=='all')       #thống kê 
#print(df.shape)     
#print(df['Lớp'].unique())
#print(df.info())
#print(df.columns)       
#print(df.Lớp.apply(lambda x: x.replace('10A1','13A0')))

#df.loc[df.Nữ.isnull(),'Nữ']='o'
#t=df.groupby('Lớp')['Password'].aggregate([min,np.mean,max])
#t=df.groupby(['Lớp','Nữ'])['Password'].mean()
#print(t.sort_values(ascending=False).head(10))              #False sort from high to low  #True sort from low to high 
#print(df.Password.value_counts().count())

#df.pivot_table(data, values=None, index=None, columns=None,
# aggfunc='mean', fill_value=None, margins=False,
# dropna=True, margins_name='All')
#titanic.pivot_table(index='sex', columns='class',
# aggfunc={'survived':sum, 'fare':'mean'})
#------------------------------------------------Example 
#sales_amounts=np.random.randint(0,20,(5,3))
#t=pd.DataFrame(sales_amounts,index=['Mon','Tues','Wed','Thurs','Fri']
#                                    ,columns=['Oreo','Cookie','Cream'])
#prices=np.array([[10,8,12]])
#butter_prices=pd.DataFrame(prices,index=['Price'],columns=['Oreo','Cookie','Cream'])
#total_prices=t.dot(butter_prices.T)
#t['Total Price']=total_prices
#print(t)
#----------------------
#name = ['Alice', 'Bob', 'Cathy', 'Doug']
#age = [25, 45, 37, 19]
#weight = [55.0, 85.5, 68.0, 61.5]
#data = np.zeros(4, dtype={'names':('name', 'age', 'weight'),'formats':('U10', 'i4', 'f8')})
##data= np.dtype([('name', 'S10'), ('age', 'i4'), ('weight', 'f8')])
##tp = np.dtype([('id', 'i8'), ('mat', 'f8', (3, 3))])
##X = np.zeros(1, dtype=tp)
# #(0, [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
#print(data.dtype)
#data['name'] = name
#data['age'] = age
#data['weight'] = weight
#print(data[data['age'] < 30]['name'])

#data_rec = data.view(np.recarray)
#print(data_rec.age)                             #quicker 

#data1=pd.DataFrame(np.random.rand(3, 2),
#           columns=['foo', 'bar'],
#            index=['a', 'b', 'c'])
#print(data1['foo'])
#ind = pd.Index([2, 3, 5, 7, 11])

#area = pd.Series({'California': 423967, 'Texas': 695662, 'New York': 141297, 'Florida': 170312, 'Illinois': 149995})
#pop = pd.Series({'California': 38332521, 'Texas': 26448193, 'New York': 19651127, 'Florida': 19552860, 'Illinois': 12882135})
#data = pd.DataFrame({'area':area, 'pop':pop})
#B = pd.DataFrame(rng.randint(0, 10, (3, 3)),columns=list('BAC'))
#print(pop/area)
#print(A.add(B,fill_value=0))

#Way 1 
#index = [('California', 2000), ('California', 2010), ('New York', 2000), ('New York', 2010), ('Texas', 2000), ('Texas', 2010)]
#populations = [33871648, 37253956,18976457, 19378102, 20851820, 25145561]
#pop = pd.Series(populations, index=index)
#index = pd.MultiIndex.from_tuples(index)
#pop = pop.reindex(index)

#Way 2 
#index = pd.MultiIndex.from_product([[2013, 2014], [1, 2]],names=['year', 'visit'])
#columns = pd.MultiIndex.from_product([['Bob', 'Guido', 'Sue'], ['HR', 'Temp']],names=['subject', 'type'])
#data = np.round(np.random.randn(4, 6), 1)
#health_data = pd.DataFrame(data, index=index, columns=columns)
#print(health_data)

#Way 3 
#data = {('California', 2000): 33871648,
#        ('California', 2010): 37253956,
#        ('Texas', 2000): 20851820,
#        ('Texas', 2010): 25145561,
#        ('New York', 2000): 18976457,
#        ('New York', 2010): 19378102}
#pd.Series(data)

#print(pop[:, 2010])

#pop_df = pop.unstack()       #convert a multiplyindexed Series into a conventionally indexed DataFrame
#pop_df.stack()               # Opposite
#pop.index.names = ['state', 'year']
#--------------------------thống kê 
#for (method, group) in planets.groupby('method'): 
#        print("{0:30s} shape={1}".format(method, group.shape))


