import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(uploaded_file):
    return pd.read_excel(uploaded_file)

def plot_correlation(df):
    corr = df.select_dtypes(include='number').corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    return fig
