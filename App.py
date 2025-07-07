import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="Prediksi Harga BBM", layout="wide")
st.title("📈 Prediksi Harga BBM ")

uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    uploaded_file.seek(0, io.SEEK_END)
    size_kb = uploaded_file.tell() / 1024

    st.success("✅ Dataset berhasil diupload!")

    st.subheader("📊 Informasi Dataset")
    col1, col2, col3 = st.columns(3)
    col1.metric("Jumlah Baris", df.shape[0])
    col2.metric("Jumlah Kolom", df.shape[1])
    col3.metric("Ukuran Dataset", f"{size_kb:.2f} KB")

    st.subheader("🔍 Tipe Data")
    st.write(df.dtypes)

    st.subheader("👀 Preview Data")
    st.dataframe(df.head())

    st.subheader("📈 Statistik Deskriptif")
    st.write(df.describe())

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        st.error("Dataset harus memiliki minimal 2 kolom numerik untuk analisis.")
    else:
        x_col = st.selectbox("Pilih kolom fitur (X)", numeric_cols)
        y_col = st.selectbox("Pilih kolom target (Y)", numeric_cols, index=1)

        X = df[[x_col]]
        y = df[y_col]
        model = LinearRegression()
        model.fit(X, y)

        nilai_input = st.number_input(f"Masukkan nilai untuk kolom '{x_col}'", value=float(X.iloc[-1][0]) + 1)
        prediksi = model.predict(np.array([[nilai_input]]))[0]

        st.subheader("📌 Hasil Prediksi")
        st.success(f"Prediksi nilai '{y_col}' saat '{x_col}' = {nilai_input}: {prediksi:,.2f}")

        st.subheader("📊 Visualisasi Data dan Prediksi")
        fig, ax = plt.subplots()
        ax.plot(X, y, marker='o', label='Data Historis')
        ax.scatter([nilai_input], [prediksi], color='red', label='Prediksi')
        ax.axvline(nilai_input, linestyle='--', color='red')
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"Regresi Linier: {y_col} terhadap {x_col}")
        ax.legend()
        st.pyplot(fig)
