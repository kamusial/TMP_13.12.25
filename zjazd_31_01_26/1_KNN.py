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

plt.scatter(5.6, 3.2, c='r')  # czerwona kropka
sns.scatterplot(data=df, x='sepallength', y='sepalwidth', hue='class')
plt.show()