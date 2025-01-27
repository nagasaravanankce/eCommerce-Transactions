import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Load the datasets
customers = pd.read_csv('Customers.csv')
transactions = pd.read_csv('Transactions.csv')

# Merge datasets to create a comprehensive customer profile
customer_data = transactions.merge(customers, on='CustomerID')

# Feature Engineering
customer_features = customer_data.groupby('CustomerID').agg(
    TotalSpending=('TotalValue', 'sum'),
    AverageTransactionValue=('TotalValue', 'mean'),
    PurchaseFrequency=('TransactionID', 'count')
).reset_index()

# Normalize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(customer_features[['TotalSpending', 'AverageTransactionValue', 'PurchaseFrequency']])

# Determine the optimal number of clusters using the Elbow Method
inertia = []
silhouette_scores = []
for n_clusters in range(2, 11):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(features_scaled)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(features_scaled, kmeans.labels_))

# Plot the Elbow Method
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(range(2, 11), inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')

# Plot Silhouette Scores
plt.subplot(1, 2, 2)
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.title('Silhouette Scores')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.show()

# Choose the optimal number of clusters (e.g., 4 based on the elbow method)
optimal_clusters = 4
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
customer_features['Cluster'] = kmeans.fit_predict(features_scaled)

# Calculate clustering metrics
db_index = davies_bouldin_score(features_scaled, customer_features['Cluster'])
silhouette_avg = silhouette_score(features_scaled, customer_features['Cluster'])

print(f'Number of Clusters Formed: {optimal_clusters}')
print(f'Davies-Bouldin Index: {db_index}')
print(f'Average Silhouette Score: {silhouette_avg}')

# Visualize the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x=customer_features['TotalSpending'], y=customer_features['AverageTransactionValue'], hue=customer_features['Cluster'], palette='viridis', s=100)
plt.title('Customer Segmentation')
plt.xlabel('Total Spending')
plt.ylabel('Average Transaction Value')
plt.legend(title='Cluster')
plt.show()

# Save the clustering results
customer_features.to_csv('Customer_Segmentation_Results.csv', index=False)
print("Customer segmentation results saved to 'Customer_Segmentation_Results.csv'.")