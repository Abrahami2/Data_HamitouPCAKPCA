import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load TrainData.csv and TestData.csv
train_data = pd.read_csv("TrainData.csv")
test_data = pd.read_csv("TestData.csv")

# Define the label column name
label_column = "Labels"
print(train_data, test_data)

# Extract labels
train_labels = train_data[label_column]
test_labels = test_data[label_column]

train_data = train_data.drop(columns=[label_column])
test_data = test_data.drop(columns=[label_column])

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(train_data, train_labels)

# Make predictions on the test data
predictions = knn.predict(test_data)

# Calculate accuracy
accuracy = accuracy_score(test_labels, predictions)

# Create a confusion matrix
conf_matrix = confusion_matrix(test_labels, predictions)

# Create a classification report
class_report = classification_report(test_labels, predictions)

# Display the accuracy
print("Accuracy:", accuracy)

# Display the confusion matrix using a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Display the classification report
print("Classification Report:")
print(class_report)