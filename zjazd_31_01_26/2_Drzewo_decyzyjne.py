import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

df = pd.read_csv('dane\\iris.csv')
print(f'Describe: {df.describe()}')
print(df)
print(f"Ile czego: {df['class'].value_counts() }")

species = {
    'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2
}
df['class_value'] = df['class'].map(species)
print(f'\nDane po przygotowaniu:\n{df.head().to_string()}')

# moja probka, moj kwiat
sample = [5.6, 3.2, 5.2, 1.45]

# plt.scatter(df.sepallength, df.sepalwidth)
# plt.show()

# plt.scatter(5.6, 3.2, c='r')  # czerwona kropka
# sns.scatterplot(data=df, x='sepallength', y='sepalwidth', hue='class')
# plt.show()
#
# sns.scatterplot(data=df, x='petallength', y='petalwidth', hue='class')
# plt.scatter(5.2, 1.45, c='r')
# plt.show()

print('Klasyfikator Decision Tree Clasifier')
# X = df.iloc[  :  ,  :4 ]   # 4 pierwsze kolumny - nie można wziąć do rysowania granic decyzyjnych
X = df.iloc[  :  ,  :2 ]
y = df.class_value
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) #, random_state=0)

from sklearn.tree import DecisionTreeClassifier
# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
print(pd.DataFrame(confusion_matrix(y_test, model.predict(X_test))))
print(pd.DataFrame(model.feature_importances_, X.columns))

# granice decyzyjne
from mlxtend.plotting import plot_decision_regions
plot_decision_regions(X.values, y.values, model)
plt.show()

