import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

# Pastikan path ke file CSV benar
file_path = "C:/Users/user/Documents/Kuliah_ses4/mesin/BBCA.JK.csv"

# Periksa apakah file ada
if not os.path.isfile(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

# Membaca data dari CSV
data = pd.read_csv(file_path)

# Memeriksa 5 baris pertama data
print(data.head())

# Memeriksa informasi tentang data
print(data.info())

# Mengonversi kolom 'Date' menjadi tipe datetime dan mengurutkan berdasarkan tanggal
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date')

# Membuat kolom 'Days' sebagai jumlah hari dari tanggal awal data
data['Days'] = (data['Date'] - data['Date'].min()).dt.days

# Plot harga penutupan saham
plt.figure(figsize=(14, 7))
plt.plot(data['Date'], data['Close'])
plt.title('Harga Penutupan Saham BCA')
plt.xlabel('Tanggal')
plt.ylabel('Harga Penutupan')
plt.show()

# Plot volume perdagangan
plt.figure(figsize=(14, 7))
plt.plot(data['Date'], data['Volume'])
plt.title('Volume Perdagangan Saham BCA')
plt.xlabel('Tanggal')
plt.ylabel('Volume Perdagangan')
plt.show()

# Statistik deskriptif
print(data.describe())

# Korelasi antara variabel
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Matriks Korelasi')
plt.show()

# Memisahkan fitur dan target
X = data[['Days', 'Volume', 'Open', 'High', 'Low']]  # Contoh fitur
y = data['Close']

# Membagi data menjadi set pelatihan dan set pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membuat dan melatih model regresi linier
model = LinearRegression()
model.fit(X_train, y_train)

# Memprediksi harga saham
y_pred = model.predict(X_test)

# Evaluasi model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Validasi silang model
cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
print(f'Cross-Validation MSE: {-cv_scores.mean()}')
