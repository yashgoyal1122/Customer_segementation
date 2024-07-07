import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Generate sample dataset
data = {
    'CustomerID': range(1, 201),
    'Age': np.random.randint(18, 70, size=200),
    'Annual Income (k$)': np.random.randint(20, 150, size=200),
    'Spending Score (1-100)': np.random.randint(1, 101, size=200)
}
df = pd.DataFrame(data)

# Preprocess data
df.drop(columns=['CustomerID'], inplace=True)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_features, columns=df.columns)

# Determine optimal number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(scaled_df)
    wcss.append(kmeans.inertia_)
plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method to Determine Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Apply K-Means clustering
kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
df['Cluster'] = kmeans.fit_predict(scaled_df)

# Visualize clusters
plt.figure(figsize=(10, 7))
sns.pairplot(df, hue='Cluster', palette='Set1', markers=['o', 's', 'D', 'P'])
plt.title('Customer Segments')
plt.show()

# 3D visualization
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter
