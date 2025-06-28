#  Task 02 – Customer Segmentation Using K-Means Clustering

Internship Domain: Machine Learning  
Company: Prodigy InfoTech  
Task: Implement K-Means clustering to group retail store customers based on purchasing behavior.

---

##  Dataset Used

- Dataset Name: Mall_Customers.csv  
- Source: [Kaggle – Customer Segmentation](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)  
- **Features Used:
  - Annual Income (k$)
  - Spending Score (1–100)

---

## What I Did

1. Loaded & explored the dataset** using pandas.
2. Selected relevant features** for clustering.
3. Standardized** the data using `StandardScaler`.
4. Used the Elbow Method to determine the optimal number of clusters.
5. Trained a K-Means model with 5 clusters.
6. Visualized the resulting clusters.
7. Saved:
   - Clustered output to `results/customer_segments.csv`
   - Elbow curve and cluster plots
   - Trained model as `kmeans_model.pkl`

---

## 📂 Folder Structure

PRODIGY_ML_02/
├── data/
│ └── Mall_Customers.csv
├── notebooks/
│ └── kmeans_clustering.py
├── models/
│ └── kmeans_model.pkl
├── results/
│ ├── elbow_curve.png
│ ├── clusters.png
│ └── customer_segments.csv
├── requirements.txt
└── README

---

##  Output Examples

###  Elbow Curve
Helps identify optimal number of clusters

###  Cluster Visualization
Groups customers based on income & spending score
