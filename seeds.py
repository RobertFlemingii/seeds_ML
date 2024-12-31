import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Define column names for the dataset
cols = ["area", "perimeter", "compactness", "length", "width", "asymmetry", "groove", "class"]

# Read the dataset from a file, providing column names and specifying a space separator
df = pd.read_csv("seeds_dataset.txt", names=cols, sep="\s+")

# Display the first few rows of the dataset
print("Dataset Preview:")
print(df.head())

#################################################
# Data Visualization
#################################################

# Pairwise scatter plots for features color-coded by 'class'
print("Visualizing pairwise feature relationships...")
for i in range(len(cols) - 1):
    for j in range(i + 1, len(cols) - 1):
        x_label = cols[i]
        y_label = cols[j]

        sns.scatterplot(x=x_label, y=y_label, data=df, hue='class', palette='tab10')
        plt.title(f"Scatter Plot: {x_label} vs {y_label}")
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend(title='Class')
        plt.show()

#################################################
# Clustering (2D)
#################################################

# Select two columns for clustering
x, y = "perimeter", "asymmetry"
X_2D = df[[x, y]].values

# Apply K-Means clustering with 3 clusters to the selected data
kmeans_2D = KMeans(n_clusters=3, random_state=42).fit(X_2D)
clusters_2D = kmeans_2D.labels_

# Create a DataFrame with cluster labels for visualization
cluster_2D_df = pd.DataFrame({x: X_2D[:, 0], y: X_2D[:, 1], "Cluster": clusters_2D})

# Visualize the data with K-Means cluster labels
sns.scatterplot(x=x, y=y, hue="Cluster", data=cluster_2D_df, palette='viridis')
plt.title(f"K-Means Clusters ({x} vs {y})")
plt.xlabel(x)
plt.ylabel(y)
plt.legend(title='Cluster')
plt.show()

# Visualize the original classes
sns.scatterplot(x=x, y=y, hue="class", data=df, palette='tab10')
plt.title(f"True Classes ({x} vs {y})")
plt.xlabel(x)
plt.ylabel(y)
plt.legend(title='Class')
plt.show()

#################################################
# Clustering (High-Dimensional)
#################################################

# Select all columns except the last one for clustering
X_highD = df[cols[:-1]].values

# Apply K-Means clustering with 3 clusters to the higher-dimensional data
kmeans_highD = KMeans(n_clusters=3, random_state=42).fit(X_highD)
clusters_highD = kmeans_highD.labels_

# Create a DataFrame with cluster labels for visualization
cluster_highD_df = pd.DataFrame(df[cols[:-1]], columns=cols[:-1])
cluster_highD_df["Cluster"] = clusters_highD

# Visualize the data in 2D using the same features as before
sns.scatterplot(x=x, y=y, hue="Cluster", data=cluster_highD_df, palette='viridis')
plt.title(f"K-Means Clusters in High Dimensions ({x} vs {y})")
plt.xlabel(x)
plt.ylabel(y)
plt.legend(title='Cluster')
plt.show()

# Visualize the original classes
sns.scatterplot(x=x, y=y, hue="class", data=df, palette='tab10')
plt.title(f"True Classes ({x} vs {y})")
plt.xlabel(x)
plt.ylabel(y)
plt.legend(title='Class')
plt.show()

#################################################
# Principal Component Analysis (PCA)
#################################################

# Apply PCA to reduce data to 2 dimensions
pca = PCA(n_components=2)
transformed_X = pca.fit_transform(X_highD)

# Create DataFrames for K-Means cluster labels and truth classes
kmeans_pca_df = pd.DataFrame({"PCA1": transformed_X[:, 0], "PCA2": transformed_X[:, 1], "Cluster": clusters_highD})
truth_pca_df = pd.DataFrame({"PCA1": transformed_X[:, 0], "PCA2": transformed_X[:, 1], "Class": df["class"]})

# Visualize the data with K-Means cluster labels in PCA space
sns.scatterplot(x="PCA1", y="PCA2", hue="Cluster", data=kmeans_pca_df, palette='viridis')
plt.title("K-Means Clusters in PCA Space")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.legend(title='Cluster')
plt.show()

# Visualize the true classes in PCA space
sns.scatterplot(x="PCA1", y="PCA2", hue="Class", data=truth_pca_df, palette='tab10')
plt.title("True Classes in PCA Space")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.legend(title='Class')
plt.show()

print("Analysis completed.")
