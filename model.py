from sklearn.linear_model import LinearRegression
import numpy as np

def train_model(df, target_col):
    X = df.drop(columns=[target_col]).select_dtypes(include=[np.number]).dropna()
    y = df[target_col].loc[X.index]
    model = LinearRegression()
    model.fit(X, y)
    return model, X.columns.tolist()

def predict_revenue(model, input_values):
    return model.predict([input_values])[0]
