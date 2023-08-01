# Projet Python - Rajeeth-A

Ce paquet est conçu pour traiter les opérations de régression linéaire de base.

# Code Files
## `ordinaryleastsquares.py`
Il contient la classe OrdinaryLeastSquares, destinée à traiter les modèles linéaires.
- **methods**
	- `__init__`
    Crée un objet OrdinaryLeastSquares vide. Les attributs sont les suivants :
		- `data_x` : Les données contenant les variables explicatives.
		- `data_y` : Les données contenant la variable d'intérêt.
		- `coeffs` : Les coefficients du modèle
		- `confiance` : Les intervalles de confiance (95%) pour les coefficients.
		- `intercept` : La présence ou non de l'ordonnée à l'origine dans le modèle.
		- `r_squared` : Le coefficient $R^2$.
		
	- `fit(self, data_x, data_y, intercept=True)`
	Calcule les coefficient du modèle à partir des données explicatives (`data_x`), des données d'intérêt (`data_y`) et de la présence ou non de l'intercept dans le modèle (`intercept)`). Modifie l'objet.
	- `get_coeffs(self, show=False)`
	Renvoie les coefficients du modèle.
	- `determination_coefficient(self)`
	Renvoie les coefficients $R^2$ du modèlet.
	-  `predict(self, new_data)`
	Utilise le modèle calculé pour prédire les valeurs de `data`.


## `ridge.py`
Elle contient la classe Ridge, identique à la classe OrdinaryLeastSquares, à l'exception de la méthode d'ajustement qui prend en compte la pénalité de crête pour calculer les coefficients. Comme elle hérite des caractéristiques de la classe OrdinaryLeastSquares, nous ne détaillerons que ce qui diffère.
- **methods**
	- `__init__`
	Crée un objet Ridge vide. Les attributs sont les suivants : 
		- `data_x` : Les données contenant les variables explicatives.
		- `data_y` : Les données contenant la variable d'intérêt.
		- `coeffs` : Les coefficients du modèle.
		- `confiance` : Les intervalles de confiance (95%) pour les coefficients.
		- `intercept` : La présence ou non de l'ordonnée à l'origine dans le modèle.
		- `r_squared` : Le coefficient $R^2$.
		- `lamb` : Le coefficient de pénalité
		
	- `fit(self, data_x, data_y, lamb, intercept=True)`
	Calcule les coefficient du modèle à partir des données explicatives (`data_x`), des données d'intérêt (`data_y`) et de la présence ou non de l'intercept dans le modèle (`intercept`) et le coefficient de pénalité (`lamb`). Modifie l'objet.