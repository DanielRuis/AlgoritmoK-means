
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#?Cargamos el csv
data = pd.read_csv('CentroComercial_Clientes.csv')


#?Seleccionar las variables a utilizar para el análisis de clúster
X = data[['Age', 'Annual Income', 'Spending Score']]

#?Determinar el número óptimo de clusters utilizando el método del codo
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)

plt.title('Método del codo')
plt.xlabel('Número de clusters')
plt.ylabel('Suma de los cuadrados de las distancias')
plt.show()

#?Aplicar el algoritmo K-means con el número óptimo de clusters
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
pred_y = kmeans.fit_predict(X)

#?Generar gráficas para visualizar los clusters identificados
plt.scatter(X.iloc[:,0], X.iloc[:,2], c=pred_y)
plt.title('Age vs Spending Score')
plt.xlabel('Edad')
plt.ylabel('Calificacion')
plt.show()

plt.scatter(X.iloc[:,1], X.iloc[:,2], c=pred_y)
plt.title('Annual Income vs Spending Score')
plt.xlabel('Ingreso Anual')
plt.ylabel('Calificaicon')
plt.show()
