# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# Load the Iris dataset
iris = load_iris()
data = iris.data
target = iris.target
feature_names = iris.feature_names

# Standardize the dataset
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Initialize K-Means with K = 3
kmeans = KMeans(n_clusters=3, random_state=42)

# Fit the model
kmeans.fit(data_scaled)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Inertia value to evaluate the performance
print("Inertia:", kmeans.inertia_)

# Visualize the first two dimensions of the data
plt.figure(figsize=(8, 6))
plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=labels, cmap='viridis', marker='o')
plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='x', label="Centroids")
plt.title('K-Means Clustering on Iris Dataset (K=3)')
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.legend()
plt.show()

# Compare the predicted cluster labels with the actual species labels
print("Cluster Labels:", labels)
print("True Labels:", target)

# Confusion matrix to analyze how the clusters map to actual labels
conf_matrix = confusion_matrix(target, labels)
print("Confusion Matrix:\n", conf_matrix)

# Elbow Method to find the optimal K
inertia_values = []
K_values = range(1, 10)

for K in K_values:
    kmeans = KMeans(n_clusters=K, random_state=42)
    kmeans.fit(data_scaled)
    inertia_values.append(kmeans.inertia_)

# Plot the Elbow Curve
plt.figure(figsize=(8, 6))
plt.plot(K_values, inertia_values, 'bo-', markersize=8)
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.show()

# Experiment with different values of K (Optional)
# Try K = 2, 4, 5 and visualize the clusters
for K in [2, 4, 5]:
    kmeans = KMeans(n_clusters=K, random_state=42)
    kmeans.fit(data_scaled)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    plt.figure(figsize=(8, 6))
    plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=labels, cmap='viridis', marker='o')
    plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='x', label="Centroids")
    plt.title(f'K-Means Clustering on Iris Dataset (K={K})')
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.legend()
    plt.show()

    # Print the inertia for each K value
    print(f"Inertia for K={K}:", kmeans.inertia_)
