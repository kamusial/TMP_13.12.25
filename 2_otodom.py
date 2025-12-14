import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('dane\\otodom.csv')
print(df.head().to_string())
print(f'Kształt danych {df.shape}')
print(df.describe().T.round(2).to_string())

# print(df.iloc[  2:7 ,3:5 ])  # wiersze od 3 do 7, kolumny od 4 do 5

# sns.heatmap(df.iloc[:,2:].corr(), annot=True)   # bez kolumny ID i bez cen
# plt.show()

# sns.histplot(df.cena)
# plt.show()
# plt.scatter(df.powierzchnia, df.cena)
# plt.show()
q1 = df.describe().T.loc['cena', '25%']  # macierz T - odwrotnie wiersze i kolumny
q3 = df.describe().loc['75%', 'cena']

df1 = df[(df.cena >= q1) & (df.cena <= q3)]
# sns.histplot(df1.cena)
# plt.show()
# plt.scatter(df1.powierzchnia, df1.cena)
# plt.show()

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

del df1['liczba_pokoi']
X = df1.iloc[:, 2:]
# X = df1.loc[:,['liczba_pieter', 'pietro', 'powierzchnia', 'rok_budowy']]
y = df1.cena

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)
print(f'Wspolczynnik kierunkowy: {model.coef_}')
print( pd.DataFrame  (model.coef_, df1.iloc[:, 2:].columns))
print(f'wyraz wolny: {model.intercept_}')
print(f'Wynik R^2 na zbiorze testowym {model.score(X_test, y_test):.4f}')
print(f'Ceny przykładowych mieszkań to: {model.predict([[4, 2, 80, 2024], [2, 8, 34, 1962]])}')

# ===========================
# DODATKOWE STATYSTYKI
# ===========================
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error

# Predykcja na zbiorze testowym
y_pred = model.predict(X_test)

# 1 Statystyki jakości modelu
r2_test = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
# rmse = mean_squared_error(y_test, y_pred, squared=False)     # nie działa ze starym sklearn
# rmse = root_mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

print("\n=== Statystyki jakości modelu ===")
print(f'R^2 (test)      : {r2_test:.4f}')
print(f'MAE (Mean Abs. Error): {mae:.2f} - czyli o ile model się myli')
print(f'MSE (Mean Sq. Error) : {mse:.2f} - czyli o ile model się myli priorytetyzując duże błędy [zł^2]')
print(f'RMSE : {rmse:.2f}')

# 2. Informacje o danych użytych w modelu
print("\n=== Statystyki dotyczące danych ===")
print(f"Liczba obserwacji przed odfiltrowaniem skrajnych cen   : {df.shape[0]}")
print(f"Liczba obserwacji po odfiltrowaniu (q1–q3)             : {df1.shape[0]}")
print(f"Liczba obserwacji w zbiorze treningowym                : {X_train.shape[0]}")
print(f"Liczba obserwacji w zbiorze testowym                   : {X_test.shape[0]}")
print(f"Średnia cena w df1 (po odfiltrowaniu skrajnych wartości): {df1.cena.mean():.2f}")
print(f"Mediana ceny w df1                                     : {df1.cena.median():.2f}")

# 3, Analiza reszt (różnica cena_rzeczywista - cena_modelu)
residuals = y_test - y_pred
print("\n=== Staytsyki reszt (błędy prognzy) ===")
print(f'Średnia reszt            : {residuals.mean():.2f}')    # najlepiej bliska 0
print(f'Odchylenie std rest      : {residuals.std():.2f}')     # typowa wielkość błędu
print(f'Minimalna reszta         : {residuals.min():.2f}')   # największe niedoszacowanie
print(f'Maksymalna reszta        : {residuals.max():.2f}')  # największe przeszacowane

# 4. WYkres histogramu reszt
sns.histplot(residuals, kde=True)
plt.title('Rozkład resz modelu (y_test - y_pred)')
plt.xlabel('Reszta [zl]')
plt.ylabel('Liczb obserwacji')
plt.show()
