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

###=========== Podstawowe miary ===========###

print('\n\nPodstawowe statystyki opisowe')
print(df[['x','y']].describe())

print('\nSkośność i kurtoza')
print(f'Skośność x: {stats.skew(df['x'])}')
print(f'Kutoza x: {stats.kurtosis(df['x'], fisher=True)}')

print('\nStatystyki pisowe w grup')
print(df.groupby('group')[['x','y']].agg(['mean', 'std', 'median']))

print('\nKorelacja')
corr, p_corr = stats.pearsonr(df['x'], df['y'])
print(f'Korelacja r = {corr:.3f}, p-value = {p_corr:.4f}')    # korelacja i czy wynik jest statystycznie istotny

### Statystyka inferencyjna ###
print('\nTest normalności Shapiro-Wilka dla x')
w_stat, p_shapiro = stats.shapiro(df['x'])
print(f'W = {w_stat:.3f}, p-value = {p_shapiro:.4f}')

print('\nTest chi-kwadrat niezależności category vs failure')
ct = pd.crosstab(df['category'], df['failure'])
print(ct)
chi2, p_chi2, dof, expecter = stats.chi2_contingency(ct)
print(f'\nChi2= {chi2:.3f}, df = {dof}, p-value = {p_chi2:.4f}')

print('\nRegresja liniowa y ~ x + group')
model= smf.ols('y ~ x + C(group)',data=df).fit()
print(model.summary())