import streamlit as st
import pandas as pd
import joblib


# Memuat model dan scaler
scaler = joblib.load("scaler_customer.joblib")
model_kmeans = joblib.load("kmeans_customer.joblib")


# Judul aplikasi
st.title("🛍️ Prediksi Cluster Pelanggan Mall")

st.write(
    "Aplikasi untuk mengelompokkan pelanggan mall "
    "berdasarkan usia, pendapatan tahunan, dan spending score."
)


# Input pengguna
age = st.number_input(
    "Age",
    min_value=1,
    max_value=100,
    value=30
)

annual_income = st.number_input(
    "Annual Income (k$)",
    min_value=0,
    value=50
)

spending_score = st.number_input(
    "Spending Score (1-100)",
    min_value=1,
    max_value=100,
    value=50
)


# Tombol prediksi
if st.button("Prediksi Cluster"):

    # Mengubah input menjadi DataFrame
    input_data = pd.DataFrame(
        [[
            age,
            annual_income,
            spending_score
        ]],
        columns=[
            "Age",
            "Annual Income (k$)",
            "Spending Score (1-100)"
        ]
    )

    # Standardisasi input
    scaled_input = scaler.transform(input_data)

    # Prediksi cluster
    cluster_result = model_kmeans.predict(scaled_input)[0]

    # Menampilkan hasil
    st.success(
        f"Pelanggan ini masuk ke dalam: Cluster {cluster_result}"
    )