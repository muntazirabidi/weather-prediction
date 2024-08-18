import numpy as np
from sklearn.metrics import mean_squared_error

def evaluate_model(y_true, y_pred, model_name="Model"):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    print(f"{model_name} - Root Mean Squared Error: {rmse:.2f}°C")
    return rmse
