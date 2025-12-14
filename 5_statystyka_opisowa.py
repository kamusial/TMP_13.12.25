import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

np.random.seed(42) # ustalamy ziarno losowe dla powtrzalności
n = 200  # zbiór danych

group = np.random.choice(['A','B'], size=n)  # 2 linie proukcyjne

# zmienna ciągła x (np. czas cyklu, temperatura)
x = np.random.normal(loc=10, scale=2, size=n)

# zmiena ciągła - koszty, powiązane z x
y = 3 * x + np.where(group == 'B', 5, 0) + np.random.normal(loc=0, scale=3, size=n)

# Zmienna kategoryczna (np. klasa jakości)
category = np.random.choice(['OK', 'Minor_defect', 'Major_defect'], size=n, p=[0.7, 0.2, 0.1])

# ZMienna binarna (czy wystąpiła awaria)
failure = np.random.binomial(n=1,p=0.25, size=n)

# Wszystko do DataFrame
df = pd.DataFrame({
    'group': group,
    'x': x,
    'y': y,
    'category': category,
    'failure': failure
})
print(df.head())
