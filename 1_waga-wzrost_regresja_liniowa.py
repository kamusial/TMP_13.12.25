import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('dane\\weight-height.csv', sep=';')
# df2 = pd.read_csv(r'dane\numbers.kamil')
# df3 = pd.read_csv('dane\\numbers.kamil')

print(df)
print(type(df))

print(df.Gender.value_counts())
# df.Height = df.Height * 2.54
df.Height *= 2.54
df.Weight /= 2.2
print('Po zmianie jednostek')
print(df)
print('Podstawowe statystyki')
# print(df.describe())
print(df.describe().round(2).T.to_string())    # T - zamiana wierszy z kolumnami

# plt.hist(df.query("Gender=='Male'").Weight, bins=30)
# plt.hist(df.query("Gender=='Female'").Weight, bins=30)
# plt.show()
#
# sns.histplot(df.query("Gender=='Male'").Weight)
# sns.histplot(df.query("Gender=='Female'").Weight)
# plt.show()

df = pd.get_dummies(df)

del (df['Gender_Male'])
df = df.rename(columns={'Gender_Female': 'Gender'})
print(df)

# df.to_csv('waga_wzrost_po_edycji.csv', index=False)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(df[['Height', 'Gender']], df['Weight'])
print(f'Wspolczynnik kierunkowy: {model.coef_}')
print(f'wyraz wolny: {model.intercept_}')

print(f'waga = wzrost * 1.06960294 + plec * -8.80805024 -102.52081454490131')
print(model.predict([[192, 0], [192, 1], [160, 1], [100, 1]]))