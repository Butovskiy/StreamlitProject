import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from model import train_model, predict_revenue
from utils import load_data, plot_correlation

st.title("Enterprise Expense Analyzer")

uploaded_file = st.file_uploader("Загрузите Excel-файл", type=["xlsx"])
if uploaded_file:
    df = load_data(uploaded_file)
    st.write("Предварительный просмотр данных:", df.head())

    st.subheader("Матрица корреляций")
    fig = plot_correlation(df)
    st.pyplot(fig)

    st.subheader("Выбор целевой переменной (например: 'Выручка')")
    target_column = st.selectbox("Целевая переменная", df.columns)

    if st.button("Обучить модель"):
        model, features = train_model(df, target_column)
        st.success("Модель обучена!")

        st.subheader("Прогнозирование")
        inputs = []
        for feature in features:
            value = st.number_input(f"{feature}", float(df[feature].min()), float(df[feature].max()), float(df[feature].mean()))
            inputs.append(value)

        if st.button("Предсказать выручку"):
            prediction = predict_revenue(model, inputs)
            st.success(f"Прогнозируемая выручка: {prediction:.2f}")
