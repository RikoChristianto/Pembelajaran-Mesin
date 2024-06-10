import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import datetime, timedelta

# Data historis harga saham
data = {
    'Date': [
        '1/2/2024', '1/3/2024', '1/4/2024', '1/5/2024', '1/8/2024', '1/9/2024', '1/10/2024',
        '1/11/2024', '1/12/2024', '1/15/2024', '1/16/2024', '1/17/2024', '1/18/2024', '1/19/2024',
        '1/22/2024', '1/23/2024', '1/24/2024', '1/25/2024', '1/26/2024', '1/29/2024', '1/30/2024',
        '1/31/2024', '2/1/2024', '2/2/2024', '2/5/2024', '2/6/2024', '2/7/2024', '2/12/2024', '2/13/2024',
        '2/15/2024', '2/16/2024', '2/19/2024', '2/20/2024', '2/21/2024', '2/22/2024', '2/23/2024',
        '2/26/2024', '2/27/2024', '2/28/2024', '2/29/2024', '3/1/2024', '3/4/2024', '3/5/2024', '3/6/2024',
        '3/7/2024', '3/8/2024', '3/13/2024', '3/14/2024', '3/15/2024', '3/18/2024', '3/19/2024', '3/20/2024',
        '3/21/2024', '3/22/2024', '3/25/2024', '3/26/2024', '3/27/2024', '3/28/2024', '4/1/2024', '4/2/2024',
        '4/3/2024', '4/4/2024', '4/5/2024', '4/16/2024', '4/17/2024', '4/18/2024', '4/19/2024', '4/22/2024',
        '4/23/2024', '4/24/2024', '4/25/2024', '4/26/2024', '4/29/2024', '4/30/2024', '5/2/2024', '5/3/2024',
        '5/6/2024'
    ],
    'Close': [
        9425, 9350, 9475, 9575, 9575, 9625, 9550, 9575, 9700, 9725, 9700, 9750, 9675, 9625, 9625, 9600, 9525, 9500,
        9350, 9550, 9650, 9550, 9700, 9700, 9575, 9625, 9700, 9800, 9725, 9850, 9950, 9875, 10025, 9975, 9875, 9825,
        9800, 9875, 10000, 9875, 9825, 9750, 9800, 9950, 10125, 10150, 10000, 10325, 10150, 10150, 10175, 10125,
        10125, 10100, 10075, 10050, 10075, 10075, 9850, 9900, 9525, 9850, 9825, 9475, 9525, 9475, 9475, 9350, 9725,
        9950, 9775, 9625, 9800, 9800, 9550, 9850, 9800
    ]
}

# Membuat DataFrame dari data
df = pd.DataFrame(data)
df['Date'] = pd.to_datetime(df['Date'])  # Mengubah kolom Date menjadi tipe datetime
df.set_index('Date', inplace=True)  # Mengatur kolom Date sebagai index DataFrame

# Memisahkan fitur (X) dan target (y)
X = df.index.values.reshape(-1, 1)  # Menggunakan tanggal sebagai fitur
y = df['Close'].values

# Membuat model regresi linier
regression_model = LinearRegression()
regression_model.fit(X, y)

# Menentukan tanggal satu minggu ke depan dari data terakhir
tanggal_terakhir = df.index[-1]
tanggal_prediksi = tanggal_terakhir + timedelta(days=7)

# Mengonversi tanggal prediksi ke dalam format yang bisa diproses oleh model
tanggal_prediksi_encoded = np.array(tanggal_prediksi.timestamp()).reshape(1, -1)

# Memprediksi harga saham satu minggu ke depan
harga_prediksi = regression_model.predict(tanggal_prediksi_encoded)

# Menampilkan hasil prediksi
print(f'Harga prediksi saham BCA untuk satu minggu ke depan ({tanggal_prediksi.date()}) adalah {harga_prediksi[0]}')
