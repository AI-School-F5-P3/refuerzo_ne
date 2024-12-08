
# Ahora, aplicamos K-Means para realizar el clustering
kmeans = KMeans(n_clusters=3, random_state=42) # Elige el número de clusters quedeseas 
kmeans.fit(datos_scaled) # Los centros de los clusters 
print("Centros de losclusters:") 
print(kmeans.cluster_centers_) # Las etiquetas de los clusters
print("Etiquetas de los clusters:") 
print(kmeans.labels_) # Puedes agregar la etiqueta
datos['Cluster'] = kmeans.labels_ # Ver las primeras filas del DataFrame con la columna de clusters 
print(datos.head())