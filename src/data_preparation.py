import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import os
import joblib

# Function to load raw data
def load_raw_data(filepath='../data/raw/jena_climate_2009_2016.csv'):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"{filepath} does not exist!")
    
    df = pd.read_csv(filepath)
    df['Date Time'] = pd.to_datetime(df['Date Time'], format='%d.%m.%Y %H:%M:%S')
    return df

# Function to handle missing data (Imputation)
def impute_missing_data(df):
    imputer = SimpleImputer(strategy='mean')
    df_imputed = pd.DataFrame(imputer.fit_transform(df.drop('Date Time', axis=1)), columns=df.columns[1:])
    df_imputed['Date Time'] = df['Date Time']
    return df_imputed

# Function for feature engineering (adding Hour, Month, Day, WeekDay)
def feature_engineering(df):
    df['Hour'] = df['Date Time'].dt.hour
    df['Month'] = df['Date Time'].dt.month
    df['Day'] = df['Date Time'].dt.day
    df['WeekDay'] = df['Date Time'].dt.dayofweek
    return df

def scale_features(df, numerical_features=['T (degC)', 'p (mbar)', 'rh (%)', 'wv (m/s)', 'max. wv (m/s)', 'wd (deg)'], scaler_file='../models/scaler.pkl'):
    scaler = StandardScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    
    # Save the scaler to a file in the 'models' folder
    joblib.dump(scaler, scaler_file)
    print(f"Scaler saved to {scaler_file}")
    
    return df


# Function to remove outliers based on temperature within each month
def remove_outliers(df):
    def outlier_filter(group):
        Q1 = group['T (degC)'].quantile(0.25)
        Q3 = group['T (degC)'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return group[(group['T (degC)'] >= lower_bound) & (group['T (degC)'] <= upper_bound)]
    
    df_clean = df.groupby('Month').apply(outlier_filter).reset_index(drop=True)
    return df_clean

# Function to save processed data
def save_processed_data(df, filepath):
    df.to_parquet(filepath, index=False)
    print(f"Data saved to {filepath}")

# Main function to run the complete pipeline
def prepare_data(raw_data_path='../data/raw/jena_climate_2009_2016.csv',
                 processed_data_path='../data/processed/'):
    
    # Create processed data directory if it doesn't exist
    os.makedirs(processed_data_path, exist_ok=True)

    # Load raw data
    df = load_raw_data(raw_data_path)
    print("Raw data loaded successfully")

    # Impute missing data
    df_imputed = impute_missing_data(df)
    print("Missing data imputed")

    # Feature engineering
    df_imputed = feature_engineering(df_imputed)
    print("Feature engineering done")

    # Scale features
    #numerical_features = ['T (degC)', 'p (mbar)', 'rh (%)', 'wv (m/s)', 'max. wv (m/s)', 'wd (deg)']
    #df_imputed = scale_features(df_imputed, numerical_features)
    #print("Numerical features scaled")

    # Remove outliers
    df_clean = remove_outliers(df_imputed)
    print(f"We removed {len(df_imputed) - len(df_clean)} unusual temperature records!")

    # Save processed data
    save_processed_data(df_imputed, os.path.join(processed_data_path, 'df_imputed.parquet'))
    save_processed_data(df_clean, os.path.join(processed_data_path, 'df_clean.parquet'))

if __name__ == "__main__":
    prepare_data()
