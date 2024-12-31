import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# define column names for the dataset
cols = ["area", "perimeter", "compactness", "length", "width", "asymmetry", "groove", "class"]

# Read the dataset from a file, providing column names and specifying a space separator
df = pd.read_csv("seeds_dataset.txt", names=cols, sep="\s+")

# Display the first few rows of the dataset
df.head()
print(df.head())

# Data Visualization Loop
# Iterate through pairs of columns for visualization
for i in range(len(cols)-1):
    for j in range(i+1, len(cols)-1):
        x_label = cols[i]
        y_label = cols[j]

        # Create a scatter plot for the selected column pair, color-coded by 'class'
        sns.scatterplot(x=x_label, y=y_label, data=df, hue='class')
        plt.show()

#################################################
#   Clustering
#################################################

# Select two columns for clustering
x = "perimeter"
y = "asymmetry"
X = df[[x, y]].values

# Apply K-Means clustering with 3 clusters to the selected data
kmeans = KMeans(n_clusters = 3).fit(X)
clusters = kmeans.labels_

# Create a DataFrame with cluster labels for visualization
cluster_df = pd.DataFrame(np.hstack((X, clusters.reshape(-1, 1))), columns=[x, y, "class"])

# Visualize the data with K-Means cluster labels
sns.scatterplot(x=x, y=y, hue='class', data=cluster_df)
plt.plot()

# Visualize the original classes
sns.scatterplot(x=x, y=y, hue='class', data=df)
plt.plot()

#################################################
# Higher Dimensions
#################################################

# Select all columns except the last one for clustering
X = df[cols[:-1]].values

# Apply K-Means clustering with 3 clusters to the higher-dimensional data
kmeans = KMeans(n_clusters = 3).fit(X)
cluster_df = pd.DataFrame(np.hstack((X, clusters.reshape(-1, 1))), columns=df.columns)

# Visualize the data with K-Means cluster labels
sns.scatterplot(x=x, y=y, hue='class', data=cluster_df)
plt.plot()

# Visualize the original classes
sns.scatterplot(x=x, y=y, hue='class', data=df)
plt.plot()

#################################################
#   PCA
#################################################

# Apply Principal Component Analysis (PCA) to reduce data to 2 dimensions
pca = PCA(n_components=2)
transformed_x = pca.fit_transform(X)

# Display the shapes of the original and transformed data
X.shape
print(X.shape)

transformed_x.shape

# Display the first 5 rows of the transformed data
transformed_x[:5]

# Create a scatter plot of the transformed data
plt.scatter(transformed_x[:,0], transformed_x[:,1])
plt.show()

# Create DataFrames for K-Means cluster labels and truth classes
kmeans_pca_df = pd.DataFrame(np.hstack((transformed_x, kmeans.labels_.reshape(-1, 1))), columns=["pca1", "pca2", "class"])
truth_pca_df = pd.DataFrame(np.hstack((transformed_x, df["class"].values.reshape(-1, 1))), columns=["pca1", "pca2", "class"])

# Visualize the data with K-Means cluster labels in PCA space
sns.scatterplot(x='pca1', y='pca2', hue='class', data=kmeans_pca_df)
plt.plot()

# Visualize the truth classes in PCA space
sns.scatterplot(x='pca1', y='pca2', hue='class', data=truth_pca_df)
plt.plot()
