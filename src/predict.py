def make_predictions(model, X_test):
    y_pred = model.predict(X_test)
    if len(y_pred.shape) > 1:
        y_pred = y_pred.flatten()
    return y_pred
