import pandas as pd

# Load the dataset
file_path = 'ObesityDataSet_raw_and_data_sinthetic.csv'
data = pd.read_csv(file_path)

# Calculate the frequency and percentage distribution for the 'NObeyesdad' column
obesity_distribution = data['NObeyesdad'].value_counts(normalize=True) * 100

# Create a DataFrame to display the result
obesity_table = pd.DataFrame({
    'Kategori Obesitas': obesity_distribution.index,
    'Jumlah Kasus (Frekuensi)': data['NObeyesdad'].value_counts(),
    'Persentase (%)': obesity_distribution
})

# Display the table to the user
print(obesity_table)

# Return the table for further use
obesity_table
