import pandas as pd

# Membaca file CSV
class_data = pd.read_csv('data/classData.csv')
silabus_data = pd.read_csv('data/filtered_silabusdata.csv')

# Menggabungkan kedua dataframe berdasarkan 'Course ID'
merged_data = pd.merge(class_data, silabus_data, on='Course ID', how='left')
# Menghapus kolom duplikat berdasarkan nama kolom
merged_data = merged_data.loc[:, ~merged_data.columns.str.contains('^Unnamed')]

# Menghapus kolom yang tidak diinginkan (misalnya kolom yang tidak relevan atau duplikat)
merged_data = merged_data.drop(columns=['Course Name_y'])

# Menampilkan hasil gabungan
print(merged_data.head())
merged_data.to_csv('merged_class_data.csv', index=False)
