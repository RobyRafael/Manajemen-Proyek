import pandas as pd

# Load the dataset
file_path = 'ObesityDataSet_raw_and_data_sinthetic.csv'
data = pd.read_csv(file_path)

# Select the relevant numeric columns for correlation analysis
numeric_columns = ['Age', 'Height', 'Weight', 'CALC', 'FAVC', 'FCVC', 'NCP', 'SCC', 'CH2O', 'FAF', 'TUE']

# Check for non-numeric values in numeric columns and display them
for col in numeric_columns:
    non_numeric = data[col].apply(lambda x: not pd.api.types.is_number(x))
    if non_numeric.any():
        print(f"Non-numeric values in column '{col}':")
        print(data[col][non_numeric])

# Convert non-numeric values in numeric columns to NaN
data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Calculate the correlation matrix using Pearson correlation method
correlation_matrix = data[numeric_columns].corr()

# Create a list for storing the correlation between pairs of attributes
correlation_pairs = []

# Iterate through the correlation matrix to create pairs of attributes and their correlation value
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        correlation_pairs.append([correlation_matrix.columns[i], correlation_matrix.columns[j], correlation_matrix.iloc[i, j]])

# Create a DataFrame to display the correlation pairs
correlation_df = pd.DataFrame(correlation_pairs, columns=['Atribut 1', 'Atribut 2', 'Nilai Korelasi'])

# Display the table to the user
print("Korelasi Antar Atribut:")
print(correlation_df)

# Return the DataFrame for further use
correlation_df
