import pandas as pd

# Load the dataset
file_path = 'ObesityDataSet_raw_and_data_sinthetic.csv'
data = pd.read_csv(file_path)

# Select the relevant numeric columns for statistical analysis
numeric_columns = ['Age', 'Height', 'Weight', 'CALC', 'FAVC', 'FCVC', 'NCP', 'SCC', 'CH2O', 'FAF', 'TUE']

# Convert non-numeric values to NaN for numeric columns
data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Generate descriptive statistics for the selected columns
desc_stats = data[numeric_columns].describe().transpose()

# Calculate additional statistics: Median and Standard Deviation
desc_stats['Median'] = data[numeric_columns].median()
desc_stats['Standard Deviation'] = data[numeric_columns].std()

# Reorganize columns for better presentation
desc_stats = desc_stats[['mean', '50%', 'min', 'max', 'std', 'Median']].rename(columns={'50%': 'Median'})

# Print the descriptive statistics
print(desc_stats)
