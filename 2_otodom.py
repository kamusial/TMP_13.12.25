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

# sns.histplot(df.cena)
# plt.show()
# plt.scatter(df.powierzchnia, df.cena)
# plt.show()
q1 = df.describe().T.loc['cena', '25%']  # macierz T - odwrotnie wiersze i kolumny
q3 = df.describe().loc['75%', 'cena']

df1 = df[(df.cena >= q1) & (df.cena <= q3)]
sns.histplot(df1.cena)
plt.show()
plt.scatter(df1.powierzchnia, df1.cena)
plt.show()

from sklearn.linear_model import LinearRegression
del df1['liczba_pokoi']
X = df1.iloc[:, 2:]
# X = df1.loc[:,['liczba_pieter', 'pietro', 'powierzchnia', 'rok_budowy']]
y = df1.cena
model = LinearRegression()
model.fit(X, y)
print(f'Wspolczynnik kierunkowy: {model.coef_}')
print( pd.DataFrame  (model.coef_, df1.iloc[:, 2:].columns))