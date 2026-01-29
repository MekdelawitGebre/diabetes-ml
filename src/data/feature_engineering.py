import pandas as pd
from src.utils.logger import get_logger

logger = get_logger("FeatureEngineering")

def add_age_bins(df):
    logger.info("Adding AgeGroup feature")
    df["AgeGroup"] = pd.cut(df["Age"], bins=[20,30,40,50,60,70,90], labels=False)
    return df

def add_bmi_features(df):
    logger.info("Adding BMI features")
    bins = [0,18.5,24.9,29.9,100]
    labels = ["Underweight","Normal","Overweight","Obese"]
    df["BMI_Category"] = pd.cut(df["BMI"], bins=bins, labels=labels)
    risk_map = {"Underweight":1,"Normal":2,"Overweight":3,"Obese":4}
    df["BMI_Risk"] = df["BMI_Category"].map(risk_map)
    return df
