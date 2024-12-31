## Clustering and Visualization with PCA

This project demonstrates clustering and data visualization techniques using the Seed Dataset, which contains measurements of different seed varieties. The primary focus is on applying K-Means clustering and visualizing results using scatter plots and Principal Component Analysis (PCA).

# Dataset

The dataset is sourced from the UCI Machine Learning Repository. It includes the following attributes:

Area

Perimeter

Compactness

Length

Width

Asymmetry

Groove Length

Class (true label for seed variety)

The data is provided in a text file (seeds_dataset.txt) with space-separated values.

# Prerequisites

Ensure the following Python libraries are installed:

pandas

numpy

matplotlib

seaborn

scikit-learn

Install them using pip if needed:

pip install pandas numpy matplotlib seaborn scikit-learn

# How It Works

Data Loading and Preprocessing:

The dataset is loaded into a pandas DataFrame.

Column names are assigned for clarity.

Data Visualization:

Pairwise scatter plots are generated for features, color-coded by the true class.

K-Means Clustering:

Clustering is performed on selected pairs of features (e.g., Perimeter vs Asymmetry).

Higher-dimensional clustering is also performed using all features except the class label.

Clustering results are visualized alongside true class labels.

Principal Component Analysis (PCA):

PCA reduces the data's dimensionality to 2D for visualization.

Scatter plots show clusters and true class labels in the PCA-transformed space.

# How to Run

Place the seeds_dataset.txt file in the same directory as the script.

Execute the Python script:

python clustering_visualization.py

The script will display various scatter plots and output key clustering results.

# Key Features

Scatter Plots: Visualize pairwise relationships between features, color-coded by class.

K-Means Clustering: Compare the clustering results with ground truth labels.

PCA: Understand how clustering performs in reduced dimensionality.

# Example Outputs

Pairwise scatter plots of features:


PCA-transformed clustering visualization:


# Notes

Clustering results may vary depending on the random initialization of centroids. For reproducibility, a random state is set in the K-Means algorithm.

PCA reduces dimensions but may lose some variance, which could impact clustering visualization accuracy.

# Acknowledgments

Dataset Source: UCI Machine Learning Repository

Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn
