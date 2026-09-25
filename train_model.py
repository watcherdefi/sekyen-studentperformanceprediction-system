import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_and_save_model():
    print("🚀 Starting Model Training Pipeline...")
    
    # 1. Load Data
    data_path = "data/student-mat.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Please check file path.")
        
    df = pd.read_csv(data_path, sep=";")
    print(f"✅ Data loaded successfully. Shape: {df.shape}")
    
    # 2. Separate Features and Target
    X = df.drop(columns=["G3"])
    y = df["G3"]
    
    # 3. Categorize Features
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    
    # 4. Define Preprocessing Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols)
        ]
    )
    
    # 5. Build Champion Pipeline (Random Forest)
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    # 6. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 7. Fit Pipeline
    print("⏳ Training Random Forest Regressor...")
    pipeline.fit(X_train, y_train)
    
    # 8. Evaluate Model
    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print("\n📊 Model Performance Metrics (Test Set):")
    print(f"   • MAE:      {mae:.4f}")
    print(f"   • RMSE:     {rmse:.4f}")
    print(f"   • R² Score: {r2:.4f}")
    
    # 9. Export Artifacts
    os.makedirs("models", exist_ok=True)
    model_path = "models/student_model.joblib"
    joblib.dump(pipeline, model_path)
    print(f"\n💾 Model pipeline saved successfully to '{model_path}'!")

if __name__ == "__main__":
    train_and_save_model()