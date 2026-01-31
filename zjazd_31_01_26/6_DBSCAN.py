# DBSCAN – Density-Based Spatial Clustering of Applications with Noise

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

iris = load_iris()
X = iris.data[:, :2]   # 2 pierwsze cechy

min_samples = 5

neighbors = NearestNeighbors(n_neighbors=min_samples)
neighbors_fit = neighbors.fit(X)
distances, indices = neighbors_fit.kneighbors(X)

# bierzemy odległość do k-tego sąsiada
distances = np.sort(distances[:, min_samples - 1])

plt.plot(distances)
plt.title('Wykres k-distance (dobór eps)')
plt.xlabel('Indeks punktu')
plt.ylabel(f'{min_samples}-ta najbliższa odległość')
plt.grid(True, alpha=0.3)
plt.show()

eps = 0.35
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
etykiety = dbscan.fit_predict(X)

unikalne_etykiety = set(etykiety)

print('\nWyniki DBSCAN:')
print(f'Liczba klastrów (bez szumu): {len(unikalne_etykiety) - (1 if -1 in etykiety else 0)}')
print(f'Liczba punktów szumu: {np.sum(etykiety == -1)}')

print('\nLiczba próbek w klastrach:')
for label in unikalne_etykiety:
    if label == -1:
        print(f'Szum: {np.sum(etykiety == label)} próbek')
    else:
        print(f'Klaster {label}: {np.sum(etykiety == label)} próbek')

kolory = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']

for label in unikalne_etykiety:
    punkty = X[etykiety == label]

    if label == -1:
        # szum
        plt.scatter(punkty[:, 0], punkty[:, 1],
                    c='black', marker='x', s=60, label='Szum')
    else:
        plt.scatter(punkty[:, 0], punkty[:, 1],
                    c=kolory[label % len(kolory)],
                    label=f'Klaster {label}',
                    alpha=0.7, s=50)

plt.title(f'Klastrowanie DBSCAN (eps={eps}, min_samples={min_samples})')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()