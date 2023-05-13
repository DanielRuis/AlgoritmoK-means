
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


#?Filtrar solo los datos de mujeres
female_data = data[data['Gender'] == 'Female']
X = female_data[['Age', 'Spending Score']].values
kmeans = KMeans(n_clusters=5, random_state=0).fit(X)
female_data['cluster'] = kmeans.labels_
best_cluster = female_data[female_data['cluster'] == female_data['cluster'].value_counts().idxmax()]
age_range = f"{best_cluster['Age'].min()}-{best_cluster['Age'].max()}"
best_cluster_data = female_data[(female_data['Age'] >= best_cluster['Age'].min()) & (female_data['Age'] <= best_cluster['Age'].max())]
plt.scatter(best_cluster_data['Age'], best_cluster_data['Spending Score'], c=best_cluster_data['cluster'])
plt.colorbar()
plt.title(f"Grupo de mujeres con mayor calificación (Edades {age_range})")
plt.xlabel('Edad')
plt.ylabel('Calificación')
plt.show()

#?Filtrar solo los datos de hombres
male_data = data[data['Gender'] == 'Male']
X = male_data[['Age', 'Spending Score']].values
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
male_data['cluster'] = kmeans.labels_
best_cluster = male_data[male_data['cluster'] == male_data['cluster'].value_counts().idxmax()]
age_range = f"{best_cluster['Age'].min()}-{best_cluster['Age'].max()}"
best_cluster_data = male_data[(male_data['Age'] >= best_cluster['Age'].min()) & (male_data['Age'] <= best_cluster['Age'].max())]
plt.scatter(best_cluster_data['Age'], best_cluster_data['Spending Score'], c=best_cluster_data['cluster'])
plt.colorbar()
plt.title(f"Grupo de hombres con mayor calificación (Edades {age_range})")
plt.xlabel('Edad')
plt.ylabel('Calificación')
plt.show()

#?Filtro para encontrar el grupo con menor calificación
min_score = data['Spending Score'].min()
min_score_group = data[data['Spending Score'] == min_score]
age_range = f"{min_score_group['Age'].min()}-{min_score_group['Age'].max()}"
min_score_group_data = data[(data['Age'] >= min_score_group['Age'].min()) & (data['Age'] <= min_score_group['Age'].max())]
plt.scatter(min_score_group_data['Age'], min_score_group_data['Spending Score'], c='red')
plt.title(f"Grupo con menor calificación (Edades {age_range})")
plt.xlabel('Edad')
plt.ylabel('Calificación')
plt.show()
data = data.drop(columns=['age_bin'])
