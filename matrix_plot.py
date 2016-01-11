import pandas as pd
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix


df = pd.read_csv('datos_peña.csv')
df1 = df.loc[:, 'Kout_60uM':'Hidrophobicity']
#print(df1)
colors = ['red', 'blue']
print(df1.shape)

scatter_matrix(df1,figsize=[20,20],marker='o',c=df.Antimicrobial.apply(lambda x:colors[x]))
plt.savefig('scatter_plot1.pdf')
#df.info()
