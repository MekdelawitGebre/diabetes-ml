import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
from src.utils.logger import get_logger

logger = get_logger("Preprocessing")

def load_data(path="data/raw/diabetes.csv"):
    logger.info(f"Loading data from {path}")
    return pd.read_csv(path)

def clean_data(df):
    logger.info("Cleaning data: replacing zeros with median and removing duplicates")
    cols_invalid_zero = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]
    df[cols_invalid_zero] = df[cols_invalid_zero].replace(0, np.nan)
    for col in cols_invalid_zero:
        df[col].fillna(df[col].median(), inplace=True)
    df.drop_duplicates(inplace=True)
    return df

def scale_features(df, feature_cols=None):
    if feature_cols is None:
        feature_cols = df.columns.drop("Outcome")
    logger.info(f"Scaling features: {feature_cols}")
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    joblib.dump(scaler, "data/processed/scaler.joblib")
    return df

def handle_imbalance(X, y):
    logger.info("Handling class imbalance with SMOTE")
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    return X_res, y_res

def train_val_test_split(X, y, test_size=0.2, val_size=0.1, seed=42):
    logger.info("Splitting data into train/validation/test sets")
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    val_relative = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_relative, stratify=y_train_val, random_state=seed
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
