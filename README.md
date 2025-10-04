# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries and load the customer dataset.

2.Select relevant features.

3.Apply the K-Means algorithm to segment customers into groups.

4.Visualize the clusters using a scatter plot. 

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: R.Sairam
RegisterNumber: 25000694
*/
```
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

data = pd.read_csv("Mall_Customers.csv")

X = data.iloc[:, [3, 4]].values

kmeans = KMeans(n_clusters=5, random_state=0)

y_kmeans = kmeans.fit_predict(X)

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=50, c='red', label='Cluster 1')

plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=50, c='blue', label='Cluster 2')

plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=50, c='green', label='Cluster 3')

plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=50, c='cyan', label='Cluster 4')

plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=50, c='magenta', label='Cluster 5')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='yellow', label='Centroids', edgecolor='black')

plt.title('Customer Segments using K-Means')

plt.xlabel('Annual Income (k$)')

plt.ylabel('Spending Score (1–100)')

plt.legend()

plt.show()

## Output: 
<img src="ex10 output.png" alt="Output" width="500">

## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
