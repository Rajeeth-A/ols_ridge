# -*- coding: utf-8 -*-
"""
Created on 2023-05-10 12:00:00

@author: Rajeeth-A
"""
import sys
import numpy as np
import pandas as pd
from mon_package.ols import OrdinaryLeastSquares as ols

class Ridge(ols):
    """
    Classe permettant de faire une regression ridge, identique à la classe OrdinaryLeastSquares,
    à l'exception de la méthode d'ajustement qui prend en compte la pénalité 
    pour calculer les coefficients. 
    Elle hérite des caractéristiques de la classe OrdinaryLeastSquares.   
    """
    def __init__(self):
        super().__init__()
        self.lam = None

    def fit(self, data_x, data_y, intercept=True, lam=None):
        self.lam = lam
        self.intercept = intercept
        data_x = data_x.copy()
        data_y = data_y.copy()
        if not isinstance(data_x, pd.DataFrame) or not isinstance(data_y, pd.DataFrame):
            sys.exit("X et y doivent être des DataFrame")
        if len(data_x) != len(data_y):
            sys.exit("Les matrices X et y n'ont pas la même taille, produit matricielle impossible")

        if self.intercept:
            data_x.insert(0, 'intercept', 1)

        identity = np.identity(data_x.shape[1])
        if self.intercept:
            identity[0, 0] = 0

        terme1 = np.linalg.inv(np.dot(data_x.transpose(), data_x) + self.lam * identity)
        terme2 = np.dot(data_x.transpose(), data_y)
        self.coeffs = np.dot(terme1, terme2)
