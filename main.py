import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the data from CSV file into a Pandas DataFrame
try:
    customer_data = pd.read_csv('Mall_Customers.csv')
except FileNotFoundError:
    print("Error: The file 'Mall_Customers.csv' was not found.")
    raise

# Display the first 5 rows of the DataFrame
print(customer_data.head())

# Display the shape of the DataFrame (rows, columns)
print(f"Data shape: {customer_data.shape}")

# Display information about the DataFrame
print(customer_data.info())

# Check for missing values in the DataFrame
print(f"Missing values per column:\n{customer_data.isnull().sum()}")

# Selecting the features 'Annual Income' and 'Spending Score'
X = customer_data.iloc[:, [3, 4]].values

# Scaling the features for better KMeans performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Finding WCSS for different numbers of clusters (1 to 10)
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Graph to determine the optimal number of clusters
sns.set()
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Point Graph')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Applying KMeans to the dataset using the optimal number of clusters
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0)
Y = kmeans.fit_predict(X_scaled)

# Plotting the clusters along with their centroids
plt.figure(figsize=(8, 8))
plt.scatter(X_scaled[Y == 0, 0], X_scaled[Y == 0, 1], s=50, c='green', label='Cluster 1')
plt.scatter(X_scaled[Y == 1, 0], X_scaled[Y == 1, 1], s=50, c='red', label='Cluster 2')
plt.scatter(X_scaled[Y == 2, 0], X_scaled[Y == 2, 1], s=50, c='yellow', label='Cluster 3')
plt.scatter(X_scaled[Y == 3, 0], X_scaled[Y == 3, 1], s=50, c='violet', label='Cluster 4')
plt.scatter(X_scaled[Y == 4, 0], X_scaled[Y == 4, 1], s=50, c='blue', label='Cluster 5')

# Plotting the centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='cyan', label='Centroids')

plt.title('Customer Groups')
plt.xlabel('Annual Income (scaled)')
plt.ylabel('Spending Score (scaled)')
plt.legend()
plt.show()
