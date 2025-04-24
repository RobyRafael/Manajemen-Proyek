import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, learning_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'ObesityDataSet_raw_and_data_sinthetic.csv'
data = pd.read_csv(file_path)

# 1. STATISTIK DESKRIPTIF DATASET
print("1. STATISTIK DESKRIPTIF DATASET")
print("="*50)

# Define numeric and categorical columns based on the dataset info
numeric_columns = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
categorical_columns = ['Gender', 'CALC', 'FAVC', 'SCC', 'SMOKE', 'family_history_with_overweight', 'CAEC', 'MTRANS', 'NObeyesdad']

# Generate descriptive statistics for the numeric columns only
desc_stats = data[numeric_columns].describe().transpose()

# Calculate additional statistics: Median
desc_stats['Median'] = data[numeric_columns].median()

# Reorganize columns for better presentation
desc_stats = desc_stats[['mean', 'min', 'max', 'std', 'Median']]

# Print and save descriptive statistics
print("Numeric Variables Statistics:")
print(desc_stats)
desc_stats.to_csv('1_statistik_deskriptif_numeric.csv')

# Visualization 1A: Numeric variables statistics
fig, axes = plt.subplots(2, 4, figsize=(16, 10))
axes = axes.ravel()

for idx, col in enumerate(numeric_columns):
    if idx < len(axes):
        axes[idx].hist(data[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
        axes[idx].set_title(f'{col} Distribution')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Frequency')
        axes[idx].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('1_statistik_deskriptif_numeric.png', dpi=300, bbox_inches='tight')
plt.close()

# Also show categorical variables distribution
print("\nCategorical Variables Distribution:")
cat_stats = {}
for col in categorical_columns:
    if col != 'NObeyesdad':  # Skip target variable for now
        value_counts = data[col].value_counts()
        cat_stats[col] = value_counts.to_dict()
        print(f"\n{col}:")
        print(value_counts)

# Save categorical statistics
cat_stats_df = pd.DataFrame.from_dict({k: pd.Series(v) for k, v in cat_stats.items()})
cat_stats_df.to_csv('1_statistik_deskriptif_categorical.csv')

# Visualization 1B: Categorical variables distribution
cat_cols = [col for col in categorical_columns if col != 'NObeyesdad']
n_cols = 3
n_rows = (len(cat_cols) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
axes = axes.ravel()

for idx, col in enumerate(cat_cols):
    if idx < len(axes):
        value_counts = data[col].value_counts()
        bars = axes[idx].bar(range(len(value_counts)), value_counts.values, alpha=0.7, edgecolor='black')
        axes[idx].set_title(f'{col} Distribution')
        axes[idx].set_xticks(range(len(value_counts)))
        axes[idx].set_xticklabels(value_counts.index, rotation=45, ha='right')
        axes[idx].set_ylabel('Count')
        axes[idx].grid(axis='y', alpha=0.3)
        
        # Add count labels on bars
        for bar, count in zip(bars, value_counts.values):
            axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                          str(count), ha='center', va='bottom')

# Hide empty subplots
for idx in range(len(cat_cols), len(axes)):
    axes[idx].set_visible(False)

plt.tight_layout()
plt.savefig('1_statistik_deskriptif_categorical.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. DISTRIBUSI KATEGORI OBESITAS
print("\n2. DISTRIBUSI KATEGORI OBESITAS")
print("="*50)

# Calculate the frequency and percentage distribution for the 'NObeyesdad' column
obesity_distribution = data['NObeyesdad'].value_counts(normalize=True) * 100

# Create a DataFrame to display the result
obesity_table = pd.DataFrame({
    'Kategori Obesitas': obesity_distribution.index,
    'Jumlah Kasus (Frekuensi)': data['NObeyesdad'].value_counts(),
    'Persentase (%)': obesity_distribution
})

# Display and save the table
print(obesity_table)
obesity_table.to_csv('2_distribusi_kategori_obesitas.csv', index=False)

# Visualization 2: Obesity distribution pie chart and bar chart
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Pie chart
colors = plt.cm.Set3(np.linspace(0, 1, len(obesity_table)))
wedges, texts, autotexts = ax1.pie(obesity_table['Persentase (%)'], 
                                  labels=obesity_table['Kategori Obesitas'],
                                  autopct='%1.1f%%',
                                  colors=colors,
                                  startangle=90)
ax1.set_title('Obesity Categories Distribution (Percentage)', fontsize=14)

# Bar chart
bars = ax2.bar(range(len(obesity_table)), obesity_table['Jumlah Kasus (Frekuensi)'], 
               color=colors, alpha=0.8, edgecolor='black')
ax2.set_xticks(range(len(obesity_table)))
ax2.set_xticklabels(obesity_table['Kategori Obesitas'], rotation=45, ha='right')
ax2.set_ylabel('Frequency', fontsize=12)
ax2.set_title('Obesity Categories Distribution (Count)', fontsize=14)
ax2.grid(axis='y', alpha=0.3)

# Add count labels on bars
for bar, count in zip(bars, obesity_table['Jumlah Kasus (Frekuensi)']):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             str(count), ha='center', va='bottom')

plt.tight_layout()
plt.savefig('2_distribusi_kategori_obesitas.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. KLASIFIKASI BERDASARKAN GENDER
print("\n3. KLASIFIKASI BERDASARKAN GENDER")
print("="*50)

# Create pivot table to show the distribution of obesity categories by gender
gender_obesity_distribution = pd.crosstab(data['Gender'], data['NObeyesdad'])

# Display and save the table
print("Klasifikasi Berdasarkan Kategori Gender:")
print(gender_obesity_distribution)
gender_obesity_distribution.to_csv('3_klasifikasi_gender.csv')

# Visualization 3: Gender vs Obesity category distribution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Stacked bar chart
gender_obesity_distribution.plot(kind='bar', stacked=True, ax=ax1, colormap='viridis', 
                               alpha=0.8, edgecolor='black')
ax1.set_xlabel('Gender', fontsize=12)
ax1.set_ylabel('Count', fontsize=12)
ax1.set_title('Obesity Categories Distribution by Gender (Stacked)', fontsize=14)
ax1.legend(title='Obesity Category', bbox_to_anchor=(1.05, 1), loc='upper left')
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)
ax1.grid(axis='y', alpha=0.3)

# Grouped bar chart
gender_obesity_distribution.plot(kind='bar', ax=ax2, colormap='viridis', alpha=0.8, edgecolor='black')
ax2.set_xlabel('Gender', fontsize=12)
ax2.set_ylabel('Count', fontsize=12)
ax2.set_title('Obesity Categories Distribution by Gender (Grouped)', fontsize=14)
ax2.legend(title='Obesity Category', bbox_to_anchor=(1.05, 1), loc='upper left')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('3_klasifikasi_gender.png', dpi=300, bbox_inches='tight')
plt.close()

# Additional visualization: Heatmap of gender vs obesity
plt.figure(figsize=(12, 8))
sns.heatmap(gender_obesity_distribution, annot=True, fmt='d', cmap='YlOrRd', 
            cbar_kws={'label': 'Count'}, linewidths=0.5)
plt.title('Heatmap: Gender vs Obesity Category', fontsize=14)
plt.xlabel('Obesity Category', fontsize=12)
plt.ylabel('Gender', fontsize=12)
plt.tight_layout()
plt.savefig('3_klasifikasi_gender_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. KORELASI ANTAR ATRIBUT
print("\n4. KORELASI ANTAR ATRIBUT")
print("="*50)

# Calculate the correlation matrix using only numeric columns
correlation_matrix = data[numeric_columns].corr()

# Create a list for storing the correlation between pairs of attributes
correlation_pairs = []

# Iterate through the correlation matrix to create pairs of attributes and their correlation value
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        correlation_pairs.append([correlation_matrix.columns[i], correlation_matrix.columns[j], correlation_matrix.iloc[i, j]])

# Create a DataFrame to display the correlation pairs
correlation_df = pd.DataFrame(correlation_pairs, columns=['Atribut 1', 'Atribut 2', 'Nilai Korelasi'])

# Display and save the table
print("Korelasi Antar Atribut (Numeric Only):")
print(correlation_df)
correlation_df.to_csv('4_korelasi_antar_atribut.csv', index=False)

# Create a correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Correlation Matrix of Numeric Variables')
plt.tight_layout()
plt.savefig('4_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. PERFORMANSI MODEL KNN
print("\n5. PERFORMANSI MODEL KNN")
print("="*50)

# Preprocessing for KNN
X = data.drop(columns=['NObeyesdad'])
y = data['NObeyesdad']

# Handle categorical features
X_encoded = pd.get_dummies(X, columns=categorical_columns[:-1], drop_first=True)  # Exclude target

# Ensure no missing values
X_encoded = X_encoded.fillna(X_encoded.median())

# Encode the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
class_names = label_encoder.classes_

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

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

# Display and save the performance table
print("Performance Model KNN:")
print(performance_df)
performance_df.to_csv('5_knn_performance_results.csv', index=False)

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
plt.savefig('5_knn_performance_plot.png', dpi=300, bbox_inches='tight')
plt.close()

# Create confusion matrix for best model
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, best_y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title(f'Confusion Matrix - KNN (K={best_k})')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('5_confusion_matrix_best_model.png', dpi=300, bbox_inches='tight')
plt.close()

# 6. EVALUASI BERDASARKAN PREPROCESSING
print("\n6. EVALUASI BERDASARKAN PREPROCESSING")
print("="*50)

# Initialize different scalers
scalers = {
    'No Scaling': None,
    'Standard Scaler': StandardScaler(),
    'MinMax Scaler': MinMaxScaler(),
    'Robust Scaler': RobustScaler()
}

# Store results
results_preprocessing = []

# Test different preprocessing methods and K values
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
        
        results_preprocessing.append({
            'Preprocessing': scaler_name,
            'K': k,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        })

# Create results DataFrame
results_df_preprocessing = pd.DataFrame(results_preprocessing)

# Find best model
best_model = results_df_preprocessing.loc[results_df_preprocessing['Accuracy'].idxmax()]
print(f"Best model: {best_model['Preprocessing']} with K={best_model['K']}")
print(f"Accuracy: {best_model['Accuracy']:.4f}")

# Create a comparison table for before/after preprocessing
before_after_comparison = pd.DataFrame({
    'Kondisi Data': ['Sebelum Preprocessing', 'Sesudah Preprocessing (Best)'],
    'Preprocessing Method': ['No Scaling', best_model['Preprocessing']],
    'K Value': [5, best_model['K']],
    'Akurasi': [
        results_df_preprocessing[(results_df_preprocessing['Preprocessing'] == 'No Scaling') & (results_df_preprocessing['K'] == 5)]['Accuracy'].values[0],
        best_model['Accuracy']
    ],
    'Precision': [
        results_df_preprocessing[(results_df_preprocessing['Preprocessing'] == 'No Scaling') & (results_df_preprocessing['K'] == 5)]['Precision'].values[0],
        best_model['Precision']
    ],
    'Recall': [
        results_df_preprocessing[(results_df_preprocessing['Preprocessing'] == 'No Scaling') & (results_df_preprocessing['K'] == 5)]['Recall'].values[0],
        best_model['Recall']
    ],
    'F1-Score': [
        results_df_preprocessing[(results_df_preprocessing['Preprocessing'] == 'No Scaling') & (results_df_preprocessing['K'] == 5)]['F1-Score'].values[0],
        best_model['F1-Score']
    ]
})

# Display and save the before/after comparison
print("\nEvaluasi Berdasarkan Preprocessing:")
print(before_after_comparison.to_string(index=False))
before_after_comparison.to_csv('6_evaluasi_preprocessing.csv', index=False)

# Save all preprocessing results
results_df_preprocessing.to_csv('6_knn_preprocessing_comparison.csv', index=False)

# Create summary visualization
plt.figure(figsize=(14, 8))
for preprocessing in results_df_preprocessing['Preprocessing'].unique():
    subset = results_df_preprocessing[results_df_preprocessing['Preprocessing'] == preprocessing]
    plt.plot(subset['K'], subset['Accuracy'], marker='o', linestyle='-', linewidth=2, 
             markersize=8, label=preprocessing)

plt.xlabel('K Value', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('KNN Performance: Different Preprocessing Methods', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(k_values)
plt.tight_layout()
plt.savefig('6_preprocessing_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Create a final summary table of all results
print("\nFINAL SUMMARY OF ALL ANALYSES")
print("="*50)
summary_files = [
    '1_statistik_deskriptif_numeric.csv',
    '1_statistik_deskriptif_categorical.csv',
    '2_distribusi_kategori_obesitas.csv',
    '3_klasifikasi_gender.csv',
    '4_korelasi_antar_atribut.csv',
    '5_knn_performance_results.csv',
    '6_evaluasi_preprocessing.csv',
    '6_knn_preprocessing_comparison.csv'
]

print("\nAll result files have been created:")
for i, file in enumerate(summary_files, 1):
    print(f"{i}. {file}")

print("\nVisualization files have been created:")
print("1. 1_statistik_deskriptif_numeric.png")
print("1. 1_statistik_deskriptif_categorical.png")
print("2. 2_distribusi_kategori_obesitas.png")
print("3. 3_klasifikasi_gender.png")
print("3. 3_klasifikasi_gender_heatmap.png")
print("4. 4_correlation_heatmap.png")
print("5. 5_knn_performance_plot.png")
print("5. 5_confusion_matrix_best_model.png")
print("6. 6_preprocessing_comparison.png")

print("\nAnalysis completed successfully!")