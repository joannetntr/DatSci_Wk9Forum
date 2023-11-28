import numpy as np
import pandas as pd

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch, DBSCAN
from sklearn.metrics import calinski_harabasz_score, silhouette_score, davies_bouldin_score

df = pd.read_csv('test.csv')
df = df.drop(columns=["Age Category", "Sex", "Qualifications", "Habitat", "Media category", "Well-being category", "anxiety category"])
print(df.head())

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

pca = PCA(n_components=2)
reduced_data = pca.fit_transform(scaled_data)

kmeans = KMeans(n_clusters=2, random_state=42)
agglomerative = AgglomerativeClustering(n_clusters=2)
dbscan = DBSCAN(eps=0.5, min_samples=5)
birch = Birch(n_clusters=2)

kmeans_labels = kmeans.fit_predict(scaled_data)
agglomerative_labels = agglomerative.fit_predict(scaled_data)
dbscan_labels = dbscan.fit_predict(scaled_data)
birch_labels = birch.fit_predict(scaled_data)

silhouette = [silhouette_score(scaled_data, predicted_labels) for predicted_labels in [kmeans_labels, agglomerative_labels, dbscan_labels, birch_labels]]
davies_bouldin = [davies_bouldin_score(scaled_data, predicted_labels) for predicted_labels in [kmeans_labels, agglomerative_labels, dbscan_labels, birch_labels]]
calinski_harabasz = [calinski_harabasz_score(scaled_data, predicted_labels) for predicted_labels in [kmeans_labels, agglomerative_labels, dbscan_labels, birch_labels]]


print("Clustering Methods     : kmeans_labels, agglomerative_lables, dbscan_labels, birch_labels")
print("Silhouette Scores      :", silhouette)
print("Davies Bouldin Indices :", davies_bouldin)
print("Calinski Harabasz Score:", calinski_harabasz)