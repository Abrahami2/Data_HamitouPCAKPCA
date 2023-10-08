import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load TrainData.csv (Replace 'TrainData.csv' with your actual data file)
train_data = pd.read_csv("TrainData.csv")

# Standardize the data
scaler = StandardScaler()
train_data_standardized = scaler.fit_transform(train_data)

# Implement PCA using scikit-learn
pca = PCA(n_components=0.95)  # Retain 95% of the variance
train_data_pca = pca.fit_transform(train_data_standardized)

# Scree Plot
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot')
plt.show()

# Biplot
plt.scatter(train_data_pca[:, 0], train_data_pca[:, 1], cmap='viridis', alpha=0.5)

# Plot principal component vectors as arrows
for i, (pc1, pc2) in enumerate(zip(pca.components_[0], pca.components_[1])):
    plt.arrow(0, 0, pc1, pc2, color='r', alpha=0.5)
    plt.text(pc1 * 1.5, pc2 * 1.5, f'Feature {i+1}', color='g', ha='center', va='center')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Biplot')
plt.show()

# Explained Variance Cumulative Plot
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance Plot')
plt.grid()
plt.show()

# Save or use train_data_pca as needed