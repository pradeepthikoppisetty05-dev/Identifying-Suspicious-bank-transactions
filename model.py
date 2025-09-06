import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

class FraudDetectionModel:
    def __init__(self, data_path='bank_transactions.csv'):
        self.data_path = data_path
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        
    def load_and_preprocess_data(self):
        """Load and preprocess the transaction data"""
        print("Loading data...")
        df = pd.read_csv(self.data_path)
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Extract additional time features
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['is_month_end'] = (df['timestamp'].dt.day > 25).astype(int)
        
        # Encode categorical variables
        categorical_cols = ['transaction_type', 'merchant', 'location']
        for col in categorical_cols:
            le = LabelEncoder()
            df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le
        
        # Select features for training
        feature_cols = [
            'amount', 'account_balance_before', 'is_weekend', 'hour_of_day',
            'amount_zscore', 'balance_ratio', 'is_high_amount', 'is_night_transaction',
            'day_of_week', 'month', 'is_month_end',
            'transaction_type_encoded', 'merchant_encoded', 'location_encoded'
        ]
        
        X = df[feature_cols]
        y = df['is_suspicious']
        
        self.feature_names = feature_cols
        
        return X, y, df
    
    def balance_dataset(self, X, y):
        """Balance the dataset using SMOTE-like approach"""
        from collections import Counter
        
        # Combine X and y
        df_combined = pd.concat([X, y], axis=1)
        
        # Separate majority and minority classes
        df_majority = df_combined[df_combined.is_suspicious == 0]
        df_minority = df_combined[df_combined.is_suspicious == 1]
        
        # Upsample minority class
        df_minority_upsampled = resample(df_minority, 
                                       replace=True,
                                       n_samples=len(df_majority)//2,
                                       random_state=42)
        
        # Combine majority class with upsampled minority class
        df_balanced = pd.concat([df_majority, df_minority_upsampled])
        
        # Separate features and target
        X_balanced = df_balanced.drop('is_suspicious', axis=1)
        y_balanced = df_balanced['is_suspicious']
        
        print(f"Original class distribution: {Counter(y)}")
        print(f"Balanced class distribution: {Counter(y_balanced)}")
        
        return X_balanced, y_balanced
    
    def train_models(self, X, y):
        """Train multiple models and select the best one"""
        print("Training models...")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(probability=True, random_state=42)
        }
        
        best_score = 0
        best_model = None
        best_model_name = None
        
        model_results = {}
        
        for name, model in models.items():
            print(f"Training {name}...")
            
            # Use scaled data for SVM and Logistic Regression
            if name in ['SVM', 'Logistic Regression']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            model_results[name] = {
                'model': model,
                'auc_score': auc_score,
                'predictions': y_pred,
                'predictions_proba': y_pred_proba,
                'classification_report': classification_report(y_test, y_pred)
            }
            
            print(f"{name} AUC Score: {auc_score:.4f}")
            
            if auc_score > best_score:
                best_score = auc_score
                best_model = model
                best_model_name = name
        
        self.model = best_model
        print(f"\nBest Model: {best_model_name} with AUC Score: {best_score:.4f}")
        
        return model_results, X_test, y_test, best_model_name
    
    def evaluate_model(self, model_results, X_test, y_test, best_model_name):
        """Evaluate the best model"""
        print(f"\nDetailed evaluation for {best_model_name}:")
        
        best_result = model_results[best_model_name]
        
        print("Classification Report:")
        print(best_result['classification_report'])
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, best_result['predictions'])
        print(cm)
        
        # Feature importance (for tree-based models)
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\nTop 10 Important Features:")
            print(feature_importance.head(10))
        
        return best_result
    
    def save_model(self, filename='fraud_detection_model.pkl'):
        """Save the trained model and preprocessors"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names
        }
        
        joblib.dump(model_data, filename)
        print(f"Model saved to {filename}")
    
    def load_model(self, filename='fraud_detection_model.pkl'):
        """Load a trained model"""
        model_data = joblib.load(filename)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.feature_names = model_data['feature_names']
        print(f"Model loaded from {filename}")
    
    def predict_transaction(self, transaction_data):
        """Predict if a single transaction is suspicious"""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Convert to DataFrame if it's a dictionary
        if isinstance(transaction_data, dict):
            df = pd.DataFrame([transaction_data])
        else:
            df = transaction_data.copy()
        
        # Preprocess the transaction data
        df = self.preprocess_single_transaction(df)
        
        # Make prediction
        prediction = self.model.predict(df)[0]
        probability = self.model.predict_proba(df)[0][1]
        
        return {
            'is_suspicious': bool(prediction),
            'suspicion_probability': float(probability),
            'risk_level': 'High' if probability > 0.7 else 'Medium' if probability > 0.3 else 'Low'
        }
    
    def preprocess_single_transaction(self, df):
        """Preprocess a single transaction for prediction"""
        # Add time features if timestamp exists
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['month'] = df['timestamp'].dt.month
            df['is_month_end'] = (df['timestamp'].dt.day > 25).astype(int)
        
        # Encode categorical variables
        categorical_cols = ['transaction_type', 'merchant', 'location']
        for col in categorical_cols:
            if col in df.columns and col in self.label_encoders:
                # Handle unknown categories
                le = self.label_encoders[col]
                df[col + '_encoded'] = df[col].apply(
                    lambda x: le.transform([str(x)])[0] if str(x) in le.classes_ else -1
                )
        
        # Add derived features
        if 'amount' in df.columns and 'account_balance_before' in df.columns:
            df['balance_ratio'] = df['amount'] / df['account_balance_before']
        
        # Select only the features used in training
        df_processed = df[self.feature_names]
        
        return df_processed
    
    def run_full_pipeline(self):
        """Run the complete training pipeline"""
        # Load and preprocess data
        X, y, df = self.load_and_preprocess_data()
        
        # Balance dataset
        X_balanced, y_balanced = self.balance_dataset(X, y)
        
        # Train models
        model_results, X_test, y_test, best_model_name = self.train_models(X_balanced, y_balanced)
        
        # Evaluate best model
        self.evaluate_model(model_results, X_test, y_test, best_model_name)
        
        # Save model
        self.save_model()
        
        return model_results

if __name__ == "__main__":
    # Initialize and run the fraud detection model
    fraud_model = FraudDetectionModel()
    
    # Run the complete pipeline
    results = fraud_model.run_full_pipeline()
    
    # Test prediction on a sample transaction
    sample_transaction = {
        'amount': 5000.0,
        'account_balance_before': 10000.0,
        'is_weekend': 0,
        'hour_of_day': 2,
        'amount_zscore': 2.5,
        'balance_ratio': 0.5,
        'is_high_amount': 1,
        'is_night_transaction': 1,
        'day_of_week': 3,
        'month': 6,
        'is_month_end': 0,
        'transaction_type_encoded': 2,
        'merchant_encoded': 1,
        'location_encoded': 5
    }
    
    prediction = fraud_model.predict_transaction(sample_transaction)
    print(f"\nSample Prediction: {prediction}")