import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import os

# STEP 1: Ensure results and models folders exist
os.makedirs('../results', exist_ok=True)
os.makedirs('../models', exist_ok=True)

# STEP 2: Load the dataset
df = pd.read_csv('../data/Mall_Customers.csv')

# STEP 3: Select features
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# STEP 4: Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# STEP 5: Elbow method to find optimal k
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# STEP 6: Save Elbow Curve plot
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o', color='teal')
plt.title('Elbow Method for Optimal Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.grid(True)
plt.savefig('../results/elbow_curve.png')
plt.close()

# STEP 7: Train final KMeans model
kmeans_final = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans_final.fit_predict(X_scaled)

# STEP 8: Add cluster labels to data
df['Cluster'] = y_kmeans

# STEP 9: Save the clustered CSV
df.to_csv('../results/customer_segments.csv', index=False)

# STEP 10: Save the trained model
joblib.dump(kmeans_final, '../models/kmeans_model.pkl')

# STEP 11: Save cluster visualization
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', palette='Set2', s=100)
plt.title('Customer Segments (K-Means)')
plt.savefig('../results/clusters.png')
plt.close()

print(" All files saved successfully.")
