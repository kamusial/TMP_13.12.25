import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('dane\\otodom.csv')
print(df.head().to_string())
print(f'Kształt danych {df.shape}')
print(df.describe().T.round(2).to_string())

# print(df.iloc[  2:7 ,3:5 ])  # wiersze od 3 do 7, kolumny od 4 do 5

sns.heatmap(df.iloc[:,2:].corr(), annot=True)   # bez kolumny ID i bez cen
plt.show()

from sklearn.linear_model import LinearRegression
X = 
y = df.cena


model.fit(X, y)