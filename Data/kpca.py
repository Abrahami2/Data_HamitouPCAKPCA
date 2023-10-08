import pandas as pd
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load TrainData.csv
train_data = pd.read_csv("TrainData.csv")

# Standardize the data
scaler = StandardScaler()
train_data_standardized = scaler.fit_transform(train_data)

# Try different kernels (e.g., 'linear', 'poly', 'sigmoid') and adjust gamma as needed
kpca = KernelPCA(kernel='linear', n_components=2)  # Adjust kernel as needed
train_data_kpca = kpca.fit_transform(train_data_standardized)

# Scatter plot of KPCA results
plt.scatter(train_data_kpca[:, 0], train_data_kpca[:, 1], cmap='viridis', alpha=0.5)
plt.xlabel('KPCA Component 1')
plt.ylabel('KPCA Component 2')
plt.title('KPCA Scatter Plot')
plt.show()