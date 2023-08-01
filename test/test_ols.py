import numpy as np
import pandas as pd
from mon_package.ols import OrdinaryLeastSquares

chemin = ("fuel2001.txt")
fuel2001 = np.genfromtxt(chemin, dtype=str)
df = pd.DataFrame(data=fuel2001[1:51,:], columns = ['Drivers', 'FuelC', 'Income', 'Miles', 'MPC', 'Pop', 'Tax', 'State'])

for col in df.columns:
    if col != 'State':
        df[col] = df[col].astype('float')

nsample = 51
Dlic = 1000 * df.Drivers / df.Pop
Fuel = 1000 * df.FuelC / df.Pop

df['Fuel'] = Fuel
df['Dlic'] = Dlic

y = np.array(df['Fuel'], dtype=float)
X = np.column_stack((df['Income'], df['Miles'], df['Tax'], df['Dlic']))

ols = OrdinaryLeastSquares()
ols.fit(X, y)
print("OLS - Coefficients:", ols.get_coeffs())

y_pred = ols.predict(X)
print("OLS - Prédictions:", y_pred)

r_squared = ols.determination_coefficient(X, y)
print("OLS - Coefficient de détermination:", r_squared)

intervalle = ols.intervalle_de_confiance(X)
print("OLS - Intervalle de confiance:", intervalle)
