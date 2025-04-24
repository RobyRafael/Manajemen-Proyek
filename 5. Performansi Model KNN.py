import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'ObesityDataSet_raw_and_data_sinthetic.csv'  # Make sure this filename is correct
data = pd.read_csv(file_path)

# Preprocessing
# Assuming 'NObeyesdad' is the target variable and rest are features
X = data.drop(columns=['NObeyesdad'])
y = data['NObeyesdad']

# Encode categorical variables in X
X = pd.get_dummies(X, drop_first=True)

# Encode the target variable if it's categorical
if y.dtype == 'object' or y.dtype.name == 'category':
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    # Store the classes for later use
    class_names = label_encoder.classes_

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data for KNN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# List of K values to evaluate
k_values = [3, 5, 7, 9, 11]

# Initialize a list to store results
results = []

# Store confusion matrices and predictions for the best model
best_accuracy = 0
best_k = 0
best_y_pred = None

# Loop over each K value to evaluate the model
for k in k_values:
    # Initialize and train the KNN classifier
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = knn.predict(X_test_scaled)
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # Append the results for this K value
    results.append([k, accuracy, precision, recall, f1])
    
    # Check if this is the best model so far
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_k = k
        best_y_pred = y_pred

# Convert results into a DataFrame
performance_df = pd.DataFrame(results, columns=['Nilai K', 'Akurasi', 'Precision', 'Recall', 'F1-Score'])

# Display the performance table
print("Performance Model KNN:")
print(performance_df)

# Save the results to CSV
performance_df.to_csv('knn_performance_results.csv', index=False)

# Visualization 1: Performance metrics across different K values
plt.figure(figsize=(12, 8))
metrics = ['Akurasi', 'Precision', 'Recall', 'F1-Score']
colors = ['blue', 'green', 'red', 'purple']

for i, metric in enumerate(metrics):
    plt.plot(performance_df['Nilai K'], performance_df[metric], marker='o', 
             linestyle='-', color=colors[i], linewidth=2, markersize=8, label=metric)

plt.xlabel('Nilai K', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.title('KNN Performance Metrics vs. Nilai K', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(k_values)
plt.ylim(0, 1.1)
plt.tight_layout()
plt.savefig('knn_performance_plot.png', dpi=300, bbox_inches='tight')
plt.close()

# Visualization 2: Bar chart of performance metrics
fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.15
x = np.arange(len(k_values))

for i, metric in enumerate(metrics):
    ax.bar(x + i*bar_width, performance_df[metric], bar_width, label=metric)

ax.set_xlabel('Nilai K')
ax.set_ylabel('Score')
ax.set_title('Comparison of Performance Metrics for Different K Values')
ax.set_xticks(x + bar_width * 1.5)
ax.set_xticklabels(k_values)
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('knn_performance_bar_chart.png', dpi=300, bbox_inches='tight')
plt.close()

# Visualization 3: Confusion Matrix for the best model
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, best_y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title(f'Confusion Matrix for KNN (K={best_k})')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig(f'confusion_matrix_k{best_k}.png', dpi=300, bbox_inches='tight')
plt.close()

# Visualization 4: Classification Report
print(f"\nClassification Report for the best model (K={best_k}):")
print(classification_report(y_test, best_y_pred, target_names=class_names))

# Create a visualization of the classification report
report = classification_report(y_test, best_y_pred, target_names=class_names, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df = report_df.iloc[:-3, :-1]  # Remove 'accuracy', 'macro avg', 'weighted avg' and 'support'

plt.figure(figsize=(10, 6))
sns.heatmap(report_df, annot=True, cmap='YlGnBu', fmt='.3f')
plt.title(f'Classification Report Heatmap (K={best_k})')
plt.tight_layout()
plt.savefig(f'classification_report_heatmap_k{best_k}.png', dpi=300, bbox_inches='tight')
plt.close()

# Summary visualization
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Line plot of metrics
for i, metric in enumerate(metrics):
    ax1.plot(performance_df['Nilai K'], performance_df[metric], marker='o', 
             linestyle='-', linewidth=2, markersize=8, label=metric)
ax1.set_xlabel('Nilai K')
ax1.set_ylabel('Score')
ax1.set_title('KNN Performance Metrics vs. Nilai K')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xticks(k_values)

# Plot 2: Bar chart
x = np.arange(len(k_values))
bar_width = 0.15
for i, metric in enumerate(metrics):
    ax2.bar(x + i*bar_width, performance_df[metric], bar_width, label=metric)
ax2.set_xlabel('Nilai K')
ax2.set_ylabel('Score')
ax2.set_title('Comparison of Performance Metrics')
ax2.set_xticks(x + bar_width * 1.5)
ax2.set_xticklabels(k_values)
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# Plot 3: Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names, ax=ax3)
ax3.set_title(f'Confusion Matrix (K={best_k})')
ax3.set_xlabel('Predicted')
ax3.set_ylabel('Actual')

# Plot 4: Classification Report Heatmap
sns.heatmap(report_df, annot=True, cmap='YlGnBu', fmt='.3f', ax=ax4)
ax4.set_title(f'Classification Report (K={best_k})')

plt.tight_layout()
plt.savefig('knn_performance_summary.png', dpi=300, bbox_inches='tight')
plt.close()

# Print summary
print(f"\nBest K value: {best_k}")
print(f"Best accuracy: {best_accuracy:.4f}")
print("\nFinal performance table:")
print(performance_df.to_string(index=False))

# Save detailed results
with open('knn_evaluation_summary.txt', 'w') as f:
    f.write("KNN Performance Evaluation Results\n")
    f.write("="*40 + "\n\n")
    f.write(f"Best K value: {best_k}\n")
    f.write(f"Best accuracy: {best_accuracy:.4f}\n\n")
    f.write("Performance across all K values:\n")
    f.write(performance_df.to_string(index=False))
    f.write("\n\nClassification Report for best model:\n")
    f.write(classification_report(y_test, best_y_pred, target_names=class_names))