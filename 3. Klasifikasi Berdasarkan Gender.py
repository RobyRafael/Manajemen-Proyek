import pandas as pd

# Load the dataset
file_path = 'ObesityDataSet_raw_and_data_sinthetic.csv'
data = pd.read_csv(file_path)

# create pivot table to show the distribution of obesity categories by gender
gender_obesity_distribution = pd.crosstab(data['Gender'], data['NObeyesdad'])

# Display the table to the user
print("Klasifikasi Berdasarkan Kategori Gender:")
print(gender_obesity_distribution)

# Return the table for further use
gender_obesity_distribution