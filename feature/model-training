from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from src.utils.logger import get_logger

logger = get_logger("ModelTraining")

def train_models(X_train, y_train):
    logger.info("Training Logistic Regression")
    lr = LogisticRegression(random_state=42)
    lr.fit(X_train, y_train)

    logger.info("Training Random Forest with GridSearchCV")
    param_grid = {'n_estimators':[100,200],'max_depth':[3,5,7]}
    rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='f1')
    rf.fit(X_train, y_train)
    best_rf = rf.best_estimator_
    logger.info(f"Best RF params: {rf.best_params_}")

    logger.info("Training XGBoost")
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb.fit(X_train, y_train)

    return {"LogisticRegression": lr, "RandomForest": best_rf, "XGBoost": xgb}
