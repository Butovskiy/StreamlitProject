# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.linear_model import LinearRegression
# from sklearn.feature_selection import f_regression

# # Функция для обработки загруженного файла
# @st.cache_data
# def load_data(file):
#     return pd.read_excel(file, sheet_name='Лист1')

# # Интерфейс Streamlit
# st.title('Анализ финансового состояния предприятия')

# # Кнопка для загрузки файла Excel
# uploaded_file = st.file_uploader("Загрузите файл Excel", type=["xlsx"])

# if uploaded_file:
#     # Загрузка данных
#     data = load_data(uploaded_file)
    
#     # Удаление ненужных столбцов
#     data = data.drop(columns=['ИНН', 'Год'])

#     # Корреляционный анализ
#     correlation_matrix = data.corr().round(1)

#     # Подготовка данных для регрессионного анализа
#     X = data.drop(columns=['выручка'])
#     y = data['выручка']

#     # Регрессионный анализ
#     model = LinearRegression()
#     model.fit(X, y)
#     coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Коэффициент'])
#     coefficients['Абсолютное значение'] = coefficients['Коэффициент'].abs()

#     # Выбор топ-5 предикторов
#     top_5_predictors = coefficients['Абсолютное значение'].sort_values(ascending=False).head(5).index
#     top_5_df = coefficients.loc[top_5_predictors]

#     # Оценка значимости предикторов
#     _, p_values = f_regression(X, y)
#     p_values_df = pd.DataFrame(p_values, index=X.columns, columns=['p_value'])
#     significant_predictors = p_values_df[p_values_df['p_value'] < 0.05]

#     tab1, tab2, tab3 = st.tabs(["Матрица корреляций", "Коэффициенты регрессионной модели", "Значимые предикторы"])

#     with tab1:
#         st.subheader('Матрица корреляций')
#         fig1, ax1 = plt.subplots(figsize=(16, 12))
#         sns.heatmap(correlation_matrix, annot=True, fmt=".1f", cmap='coolwarm', ax=ax1, annot_kws={"size": 8})
#         st.pyplot(fig1)

#     with tab2:
#         st.subheader('Коэффициенты регрессионной модели')
#         fig2, ax2 = plt.subplots(figsize=(16, 12))
#         sns.barplot(x=coefficients.index, y=coefficients['Коэффициент'], ax=ax2)
#         ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90)
#         st.pyplot(fig2)

#     with tab3:
#         st.subheader('Значимые предикторы')
#         fig3, ax3 = plt.subplots(figsize=(16, 12))
#         sns.barplot(x=significant_predictors.index, y=significant_predictors['p_value'], ax=ax3)
#         ax3.set_xticklabels(ax3.get_xticklabels(), rotation=90)
#         st.pyplot(fig3)

#     st.subheader('Топ-5 предикторов')
#     st.write(top_5_df)
# else:
#     st.info('Пожалуйста, загрузите файл Excel для анализа.')



import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression
import os

# Функция для копирования и сохранения файла в корень программы
def save_uploadedfile(uploadedfile):
    try:
        with open(os.path.join("uploaded_file.xlsx"), "wb") as f:
            f.write(uploadedfile.getbuffer())
        return st.success("Файл загружен: {}".format(uploadedfile.name))
    except Exception as e:
        return st.error(f"Ошибка при загрузке файла: {e}")

# Функция для обработки загруженного файла
@st.cache_data
def load_data(file_path):
    try:
        return pd.read_excel(file_path, sheet_name='Лист1')
    except Exception as e:
        st.error(f"Ошибка при чтении файла: {e}")
        return None

# Интерфейс Streamlit
st.title('Анализ финансового состояния предприятия')

# Пояснительное сообщение
st.write("""
Пожалуйста, загрузите файл Excel с вашего компьютера для проведения анализа. 
Ваши данные будут сохранены локально и не будут передаваться на сервер.
""")

# Кнопка для загрузки файла Excel
uploaded_file = st.file_uploader("Загрузите файл Excel", type=["xlsx"])

if uploaded_file:
    save_uploadedfile(uploaded_file)
    data = load_data("uploaded_file.xlsx")
    
    if data is not None:
        # Удаление ненужных столбцов
        if 'ИНН' in data.columns and 'Год' in data.columns:
            data = data.drop(columns=['ИНН', 'Год'])

        # Корреляционный анализ
        correlation_matrix = data.corr().round(1)

        # Подготовка данных для регрессионного анализа
        X = data.drop(columns=['выручка'])
        y = data['выручка']

        # Регрессионный анализ
        model = LinearRegression()
        model.fit(X, y)
        coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Коэффициент'])
        coefficients['Абсолютное значение'] = coefficients['Коэффициент'].abs()

        # Выбор топ-5 предикторов
        top_5_predictors = coefficients['Абсолютное значение'].sort_values(ascending=False).head(5).index
        top_5_df = coefficients.loc[top_5_predictors]

        # Оценка значимости предикторов
        _, p_values = f_regression(X, y)
        p_values_df = pd.DataFrame(p_values, index=X.columns, columns=['p_value'])
        significant_predictors = p_values_df[p_values_df['p_value'] < 0.05]

        tab1, tab2, tab3 = st.tabs(["Матрица корреляций", "Коэффициенты регрессионной модели", "Значимые предикторы"])

        with tab1:
            st.subheader('Матрица корреляций')
            fig1, ax1 = plt.subplots(figsize=(16, 12))
            sns.heatmap(correlation_matrix, annot=True, fmt=".1f", cmap='coolwarm', ax=ax1, annot_kws={"size": 8})
            st.pyplot(fig1)

        with tab2:
            st.subheader('Коэффициенты регрессионной модели')
            fig2, ax2 = plt.subplots(figsize=(16, 12))
            sns.barplot(x=coefficients.index, y=coefficients['Коэффициент'], ax=ax2)
            ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90)
            st.pyplot(fig2)

        with tab3:
            st.subheader('Значимые предикторы')
            fig3, ax3 = plt.subplots(figsize=(16, 12))
            sns.barplot(x=significant_predictors.index, y=significant_predictors['p_value'], ax=ax3)
            ax3.set_xticklabels(ax3.get_xticklabels(), rotation=90)
            st.pyplot(fig3)

        st.subheader('Топ-5 предикторов')
        st.write(top_5_df)
else:
    st.info('Пожалуйста, загрузите файл Excel для анализа.')
