# -*- coding: utf-8 -*-
"""
Created on 2023-05-10 12:00:00

@author: Rajeeth-A
"""
import sys
import numpy as np
import pandas as pd

class OrdinaryLeastSquares:
    """
    Classe permettant de calculer un modèle de régression linéaire 
    à partir de la méthode des moindres carrés ordinaires.
    """
    def __init__(self):
        """
        Constructeur de la classe OrdinaryLeastSquares.
        """
        self.data_x = None
        self.data_y = None
        self.coeffs = None
        self.r_squared = None
        self.intercept = None
        self.prediction = None
        self.confiance = None

    def fit(self, data_x, data_y, intercept=True):
        """_summary_
        Calcule les coefficient du modèle à partir des (`data_x`), (`data_y`) 
        et de la présence ou non de l'intercept dans le modèle (`intercept)`).
        Modifie l'objet.
        
        Args:
            data_x (array): données explicatives
            data_y (array): données d'intérêt/ de sortie.
            intercept (bool, optional):  Indique si l'intercept doit être inclus dans le modèle. 
                                        Par défaut, True.
        Returns:
            None
        """
        self.intercept = intercept
        if len(data_x) != len(data_y):
            sys.exit("Les matrices X et y n'ont pas la même taille, produit matricielle impossible")

        data_x = np.hstack((np.ones((data_x.shape[0], 1)), data_x))
        terme1 = np.linalg.inv(np.dot(data_x.transpose(), data_x))
        terme2 = np.dot(data_x.transpose(), data_y)
        self.coeffs = np.dot(terme1, terme2)

    def predict(self, data_x):
        """_summary_
        Utilise le modèle calculé pour prédire les valeurs de `data`.
        Args:
            data_x (_df_): Les données explicatives 
            pour lesquelles les prédictions doivent être effectuées.

        Returns:
            _df_: Les valeurs prédites à partir du modèle de régression linéaire.
        """
        data = data_x.copy()
        if self.coeffs is None:
            sys.exit("Le modèle n'a pas encore été ajusté avec les données")
        else:
            if isinstance(data, pd.DataFrame):
                if self.intercept:
                    col_intercept = np.ones((data.shape[0], 1))
                    data.insert(0, 'intercept', col_intercept)
                return np.dot(data.values, self.coeffs)
            if isinstance(data, (np.matrix, np.ndarray)):
                if self.intercept:
                    col_intercept = np.ones((data.shape[0], 1))
                    data = np.hstack((col_intercept, data))
                return np.dot(data, self.coeffs)
        self.prediction = np.dot(data, self.coeffs)
        return self.prediction

    def get_coeffs(self):
        """_summary_
        Renvoie les coefficients du modèle.
        
        Returns:
            array: Les coefficients du modèle
        """
        if self.coeffs is None:
            sys.exit("Le modèle n'a pas encore été ajusté avec les données")
        return self.coeffs

    def determination_coefficient(self, data_x, data_y):
        """_summary_
    
        Calcule le coefficient de détermination (R²) du modèle.

        Args:
            X (array-like): Les variables indépendantes.
            data_y (array-like): Les valeurs observées de la variable dépendante.

        Returns:
            float: Le coefficient de détermination (R²).
    
        """
        if self.coeffs is None:
            sys.exit("Le modèle n'a pas encore été ajusté avec les données")

        data_y_pred = self.predict(data_x)
        ssr = np.sum((data_y - data_y_pred) ** 2)
        sst = np.sum((data_y - np.mean(data_y)) ** 2)
        self.r_squared = 1 - (ssr / sst)
        return self.r_squared

    def intervalle_de_confiance(self, data_x):
        """_summary_
        Calcule l'intervalle de confiance pour les prédictions du modèle.

        Args:
            X (array): Les variables indépendantes.

        Returns:
            list: Liste contenant les bornes inférieure et supérieure de l'intervalle de confiance.
        """
        if self.coeffs is None:
            sys.exit("Le modèle n'a pas encore été ajusté avec les données")
        if isinstance(data_x, pd.DataFrame):
            sys.exit("Calculer l'inverse d'un DataFrame n'a aucun sens en language mathématique.")
        if not isinstance(data_x, np.ndarray) or isinstance(data_x, np.matrix):
            sys.exit("X doit être une Matrice pour calculer l'intervalle de confiance")
        if self.intercept:
            new_data_x = np.hstack((np.ones((data_x.shape[0], 1)), data_x))
        borne_sup = self.coeffs + 1.96 * (
            np.sqrt(np.diag(np.linalg.inv(np.dot(new_data_x.transpose(), new_data_x))))
        )
        borne_sup = self.coeffs - 1.96 * (
            np.sqrt(np.diag(np.linalg.inv(np.dot(new_data_x.transpose(), new_data_x))))
        )
        self.confiance = [borne_sup, borne_sup]
        return self.confiance
