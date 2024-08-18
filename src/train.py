import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler

# File paths
MODEL_DIR = '../models'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RF_MODEL_FILE_CLEAN = os.path.join(MODEL_DIR, 'random_forest_model_clean.pkl')
RF_MODEL_FILE_IMPUTED = os.path.join(MODEL_DIR, 'random_forest_model_imputed.pkl')
RF_MODEL_FILE_RAW = os.path.join(MODEL_DIR, 'random_forest_model.pkl')
IMPUTED_NN_MODEL_FILE = os.path.join(BASE_DIR,  'neural_network_model_imputed.keras')
CLEAN_NN_MODEL_FILE = os.path.join(BASE_DIR,  'neural_network_model_clean.keras')
RAW_NN_MODEL_FILE = os.path.join(BASE_DIR,  'neural_network_model.keras')

# Ensure that the model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

def train_random_forest(X_train_scaled, y_train, dataset_type):
    """Train or load a Random Forest model based on dataset type (imputed or clean)."""
    
    if dataset_type == "raw":
        rf_model_file = RF_MODEL_FILE_RAW
    elif dataset_type == "imputed":
        rf_model_file = RF_MODEL_FILE_IMPUTED
    else:
        rf_model_file = RF_MODEL_FILE_CLEAN


    if os.path.exists(rf_model_file):
        try:
            with open(rf_model_file, 'rb') as file:
                rf_model = joblib.load(file)
            print(f"Random Forest model loaded from {rf_model_file}")
        except Exception as e:
            print(f"Error loading Random Forest model: {e}")
            rf_model = None
    else:
        rf_model = None

    if rf_model is None:
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train_scaled, y_train)
        with open(rf_model_file, 'wb') as file:
            joblib.dump(rf_model, file)
        print(f"Random Forest model trained and saved to {rf_model_file}")
    
    return rf_model

def train_neural_network(X_train_scaled, y_train, dataset_type):
    """Train or load a Neural Network model based on dataset type (imputed or clean)."""
    
    if dataset_type == "raw":
        nn_model_file = RAW_NN_MODEL_FILE
    elif dataset_type == "imputed":
        nn_model_file = IMPUTED_NN_MODEL_FILE
    else:
        nn_model_file = CLEAN_NN_MODEL_FILE

    if os.path.exists(nn_model_file):
        try:
            nn_model = load_model(nn_model_file)
            print(f"Neural Network model loaded from {nn_model_file}")
        except Exception as e:
            print(f"Error loading Neural Network model: {e}")
            nn_model = None
    else:
        nn_model = None

    if nn_model is None:
        nn_model = Sequential([
            Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(1)  # Assuming regression
        ])
        nn_model.compile(optimizer='adam', loss='mean_squared_error')

        # Adding EarlyStopping and ModelCheckpoint for better training
        early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
        checkpoint = ModelCheckpoint(nn_model_file, save_best_only=False)  # Save every epoch for testing

        print(f"Training model and saving to {nn_model_file}...")
        nn_model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, verbose=1, 
                     callbacks=[early_stopping, checkpoint])

        # Explicitly save the model after training
        try:
            print(f"Saving model to: {os.path.abspath(nn_model_file)}")
            nn_model.save(nn_model_file)
            print(f"Saving model to: {os.path.abspath(nn_model_file)}")
            print(f"✅ Model explicitly saved to {nn_model_file}")
        except Exception as e:
            print(f"❌ Error saving model: {e}")

    return nn_model

def scale_and_save_data(X_train, X_test, scaler_file):
    """Scale data and save the scaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Ensure the directory exists
    scaler_dir = os.path.dirname(scaler_file)
    if not os.path.exists(scaler_dir):
        os.makedirs(scaler_dir)
    
    # Save the scaler
    joblib.dump(scaler, scaler_file)
    print(f"Scaler saved to {scaler_file}")
    
    return X_train_scaled, X_test_scaled



