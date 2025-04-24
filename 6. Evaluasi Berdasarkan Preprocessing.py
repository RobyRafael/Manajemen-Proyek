import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import learning_curve

# Load the dataset
file_path = 'ObesityDataSet_raw_and_data_sinthetic.csv'
data = pd.read_csv(file_path)

# Preprocess data
# Handle categorical data for classification
label_encoder = LabelEncoder()
data['NObeyesdad'] = label_encoder.fit_transform(data['NObeyesdad'])
class_names = label_encoder.classes_

# Handle categorical features properly
categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
if 'NObeyesdad' in categorical_columns:
    categorical_columns.remove('NObeyesdad')

# One-hot encode categorical features
data_encoded = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

# Select numeric and encoded features
feature_columns = [col for col in data_encoded.columns if col != 'NObeyesdad']
X = data_encoded[feature_columns]
y = data_encoded['NObeyesdad']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize different scalers
scalers = {
    'No Scaling': None,
    'Standard Scaler': StandardScaler(),
    'MinMax Scaler': MinMaxScaler(),
    'Robust Scaler': RobustScaler()
}

# Store results
results = []

# Test different preprocessing methods and K values
k_values = [3, 5, 7, 9, 11]

for scaler_name, scaler in scalers.items():
    for k in k_values:
        # Apply preprocessing if scaler is not None
        if scaler is not None:
            X_train_processed = scaler.fit_transform(X_train)
            X_test_processed = scaler.transform(X_test)
        else:
            X_train_processed = X_train.values
            X_test_processed = X_test.values
        
        # Train KNN model
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_processed, y_train)
        
        # Make predictions
        y_pred = knn.predict(X_test_processed)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Perform cross-validation
        cv_scores = cross_val_score(knn, X_train_processed, y_train, cv=5)
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        results.append({
            'Preprocessing': scaler_name,
            'K': k,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'CV Mean': cv_mean,
            'CV Std': cv_std
        })

# Create results DataFrame
results_df = pd.DataFrame(results)

# Find best model
best_model = results_df.loc[results_df['Accuracy'].idxmax()]
print(f"Best model: {best_model['Preprocessing']} with K={best_model['K']}")
print(f"Accuracy: {best_model['Accuracy']:.4f}")

# Visualization 1: Performance comparison across preprocessing methods
plt.figure(figsize=(14, 8))
for preprocessing in results_df['Preprocessing'].unique():
    subset = results_df[results_df['Preprocessing'] == preprocessing]
    plt.plot(subset['K'], subset['Accuracy'], marker='o', linestyle='-', linewidth=2, 
             markersize=8, label=preprocessing)

plt.xlabel('K Value', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('KNN Performance: Different Preprocessing Methods', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(k_values)
plt.tight_layout()
plt.savefig('preprocessing_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Visualization 2: Bar chart comparing all metrics
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.ravel()

for idx, metric in enumerate(metrics):
    ax = axes[idx]
    pivot_data = results_df.pivot(index='K', columns='Preprocessing', values=metric)
    pivot_data.plot(kind='bar', ax=ax, width=0.8)
    ax.set_title(f'{metric} by K Value and Preprocessing Method')
    ax.set_xlabel('K Value')
    ax.set_ylabel(metric)
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Visualization 3: Heatmap of all results
plt.figure(figsize=(12, 10))
pivot_accuracy = results_df.pivot(index='K', columns='Preprocessing', values='Accuracy')
sns.heatmap(pivot_accuracy, annot=True, fmt='.3f', cmap='YlGnBu', cbar_kws={'label': 'Accuracy'})
plt.title('Accuracy Heatmap: K vs Preprocessing Method')
plt.tight_layout()
plt.savefig('accuracy_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# Create confusion matrix for best model
best_scaler_name = best_model['Preprocessing']
best_k = int(best_model['K'])

if best_scaler_name != 'No Scaling':
    scaler = scalers[best_scaler_name]
    X_train_best = scaler.fit_transform(X_train)
    X_test_best = scaler.transform(X_test)
else:
    X_train_best = X_train.values
    X_test_best = X_test.values

knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train_best, y_train)
y_pred_best = knn_best.predict(X_test_best)

# Confusion Matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title(f'Confusion Matrix - {best_scaler_name} with K={best_k}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('best_model_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# Learning Curve
train_sizes, train_scores, test_scores = learning_curve(
    knn_best, X_train_best, y_train, cv=5, n_jobs=-1, 
    train_sizes=np.linspace(0.1, 1.0, 10))

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
plt.xlabel("Training examples")
plt.ylabel("Score")
plt.title(f"Learning Curves - {best_scaler_name} with K={best_k}")
plt.legend(loc="best")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('learning_curve.png', dpi=300, bbox_inches='tight')
plt.close()

# Create summary table for best preprocessing method at each K
summary_table = results_df.loc[results_df.groupby('K')['Accuracy'].idxmax()]
summary_table = summary_table[['K', 'Preprocessing', 'Accuracy', 'Precision', 'Recall', 'F1-Score']]

# Display results
print("\nBest Preprocessing Method for Each K Value:")
print(summary_table.to_string(index=False))

# Save detailed results
results_df.to_csv('knn_preprocessing_comparison.csv', index=False)

# Create a comparison table for before/after preprocessing (best preprocessing vs no preprocessing)
before_after_comparison = pd.DataFrame({
    'Kondisi Data': ['Sebelum Preprocessing', 'Sesudah Preprocessing (Best)'],
    'Preprocessing Method': ['No Scaling', best_model['Preprocessing']],
    'K Value': [5, best_model['K']],
    'Akurasi': [
        results_df[(results_df['Preprocessing'] == 'No Scaling') & (results_df['K'] == 5)]['Accuracy'].values[0],
        best_model['Accuracy']
    ],
    'Precision': [
        results_df[(results_df['Preprocessing'] == 'No Scaling') & (results_df['K'] == 5)]['Precision'].values[0],
        best_model['Precision']
    ],
    'Recall': [
        results_df[(results_df['Preprocessing'] == 'No Scaling') & (results_df['K'] == 5)]['Recall'].values[0],
        best_model['Recall']
    ],
    'F1-Score': [
        results_df[(results_df['Preprocessing'] == 'No Scaling') & (results_df['K'] == 5)]['F1-Score'].values[0],
        best_model['F1-Score']
    ]
})

# Display the before/after comparison
print("\nEvaluasi Berdasarkan Preprocessing:")
print(before_after_comparison.to_string(index=False))

# Create a final summary visualization
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Accuracy comparison
for preprocessing in results_df['Preprocessing'].unique():
    subset = results_df[results_df['Preprocessing'] == preprocessing]
    ax1.plot(subset['K'], subset['Accuracy'], marker='o', linestyle='-', linewidth=2, 
             markersize=8, label=preprocessing)
ax1.set_xlabel('K Value')
ax1.set_ylabel('Accuracy')
ax1.set_title('KNN Performance: Different Preprocessing Methods')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Best preprocessing for each K
best_methods = results_df.loc[results_df.groupby('K')['Accuracy'].idxmax()]
ax2.bar(best_methods['K'], best_methods['Accuracy'])
for i, row in best_methods.iterrows():
    ax2.text(row['K'], row['Accuracy'], row['Preprocessing'], ha='center', va='bottom')
ax2.set_xlabel('K Value')
ax2.set_ylabel('Best Accuracy')
ax2.set_title('Best Preprocessing Method for Each K')
ax2.grid(axis='y', alpha=0.3)

# Plot 3: Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names, ax=ax3)
ax3.set_title(f'Confusion Matrix - {best_scaler_name} with K={best_k}')
ax3.set_xlabel('Predicted')
ax3.set_ylabel('Actual')

# Plot 4: Before/After comparison bar chart
metrics = ['Akurasi', 'Precision', 'Recall', 'F1-Score']
x = np.arange(len(metrics))
width = 0.35
before_values = before_after_comparison.iloc[0][3:].values
after_values = before_after_comparison.iloc[1][3:].values

ax4.bar(x - width/2, before_values, width, label='Before Preprocessing')
ax4.bar(x + width/2, after_values, width, label='After Preprocessing')
ax4.set_xlabel('Metrics')
ax4.set_ylabel('Score')
ax4.set_title('Before vs After Preprocessing Comparison')
ax4.set_xticks(x)
ax4.set_xticklabels(metrics)
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('knn_analysis_summary.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nAnalysis complete. Files saved:")
print("1. preprocessing_comparison.png")
print("2. metrics_comparison.png")
print("3. accuracy_heatmap.png")
print("4. best_model_confusion_matrix.png")
print("5. learning_curve.png")
print("6. knn_analysis_summary.png")
print("7. knn_preprocessing_comparison.csv")
print("8. knn_analysis_results.xlsx")