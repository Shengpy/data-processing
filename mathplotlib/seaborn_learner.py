import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_excel('S.xls')
#x = np.arange(5)
#y=np.empty(5)
#np.multiply(x, 10, out=y)
#y = np.zeros(10)
#np.power(2, x, out=y[::2])
#print(y)

#x = np.arange(1, 6)
#print(np.multiply.reduce(x))
#print( np.add.accumulate(x))
#print(np.add.outer(x, x))

#def make_df(cols, ind):
#    data = {c: [str(c) + str(i) for i in ind] for c in cols}
#    return pd.DataFrame(data, ind)
#data=make_df('ABC', range(3))
#print(data)

#merged.loc[merged['state'].isnull(), 'state/region'].unique()                  #find out the null values in state/region
#merged.loc[merged['state/region'] == 'USA', 'state'] = 'United States'

