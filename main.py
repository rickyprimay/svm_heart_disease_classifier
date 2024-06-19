import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump, load
import os

# Fungsi untuk melatih model SVM
def train_model(df):
    X = df.drop(columns='target')
    y = df['target']
    
    # Pembagian data menjadi data latih dan data uji
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Standardisasi fitur
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Inisialisasi dan pelatihan model SVM
    svm_model = SVC(kernel='linear', probability=True)
    svm_model.fit(X_train, y_train)
    
    # Evaluasi model
    y_pred = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Menyimpan model dan scaler
    dump(svm_model, 'svm_model.joblib')
    dump(scaler, 'scaler.joblib')
    
    return accuracy

# Fungsi untuk memuat model dan scaler
def load_model():
    if os.path.exists('svm_model.joblib') and os.path.exists('scaler.joblib'):
        svm_model = load('svm_model.joblib')
        scaler = load('scaler.joblib')
        return svm_model, scaler
    else:
        return None, None

# Fungsi untuk melakukan prediksi
def make_prediction(model, scaler, input_data):
    input_data = np.array(input_data).reshape(1, -1)
    input_data = scaler.transform(input_data)
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)
    return prediction, prediction_proba

# Fungsi utama
def main():
    st.title("Klasifikasi Penyakit Jantung Menggunakan SVM")

    df = pd.read_csv('heart.csv')
    train_model(df)
    
    st.write("Matrix Korelasi:")
    matrix = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Matrix Korelasi')
    st.pyplot(plt)
    
    jenis_kelamin_label = {
        0: "Perempuan",
        1: "Laki Laki"
    }

    model, scaler = load_model()
    st.sidebar.header("Prediksi Penyakit Jantung")
    age = st.sidebar.number_input("Usia", min_value=0, max_value=120, value=30)
    sex = st.sidebar.selectbox("Jenis Kelamin", options=[0, 1], format_func=lambda x: jenis_kelamin_label[x])
    cp = st.sidebar.selectbox("Jenis Nyeri Dada (cp)", [0, 1, 2, 3])
    trestbps = st.sidebar.number_input("Tekanan Darah Istirahat (trestbps)", min_value=0, max_value=300, value=120)
    chol = st.sidebar.number_input("Kolesterol Serum (chol)", min_value=0, max_value=600, value=200)
    fbs = st.sidebar.selectbox("Gula Darah Puasa > 120 mg/dl (fbs)", [0, 1])
    restecg = st.sidebar.selectbox("Hasil Elektrokardiografi Istirahat (restecg)", [0, 1, 2])
    thalach = st.sidebar.number_input("Denyut Jantung Maksimum Terpencapai (thalach)", min_value=0, max_value=250, value=150)
    exang = st.sidebar.selectbox("Angina yang Diinduksi Latihan (exang)", [0, 1])
    oldpeak = st.sidebar.number_input("Depresi ST yang Diinduksi Latihan (oldpeak)", min_value=0.0, max_value=10.0, value=1.0)
    slope = st.sidebar.selectbox("Kemiringan Segmen ST Latihan Puncak (slope)", [0, 1, 2])
    ca = st.sidebar.selectbox("Jumlah Pembuluh Darah Utama (ca)", [0, 1, 2, 3, 4])
    thal = st.sidebar.selectbox("Talasemia (thal)", [0, 1, 2, 3])
    
    input_data = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
    
    if st.sidebar.button("Prediksi"):
        prediction, prediction_proba = make_prediction(model, scaler, input_data)
        st.sidebar.write(f'Prediksi: {"Tidak Ada Penyakit Jantung" if prediction[0] == 0 else "Terindikasi Penyakit Jantung"}')
        st.sidebar.write(f'Probabilitas: {prediction_proba[0][prediction[0]]:.2f}')

if __name__ == "__main__":
    main()
