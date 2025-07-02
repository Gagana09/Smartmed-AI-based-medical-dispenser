import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import pickle
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetProcessor:
    def __init__(self, dataset_path="New IDP Dataset .xlsx"):
        self.dataset_path = dataset_path
        self.df = None
        self.model = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.target_column = 'OTC/Doc'
        
    def load_dataset(self):
        """Load the Excel dataset"""
        try:
            self.df = pd.read_excel(self.dataset_path)
            logger.info(f"Dataset loaded successfully. Shape: {self.df.shape}")
            logger.info(f"Columns: {list(self.df.columns)}")
            return True
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return False
    
    def preprocess_data(self):
        """Preprocess the dataset for training"""
        if self.df is None:
            logger.error("Dataset not loaded. Call load_dataset() first.")
            return False
        
        # Create a copy for preprocessing
        df_processed = self.df.copy()
        
        # Handle missing values
        df_processed = df_processed.fillna('No')
        
        # Encode categorical variables
        categorical_columns = ['Gender', 'Age', 'Weight', 'Fever', 'Common Cold', 'Cough', 
                             'Body Pain', 'Headache', 'Menstrual Cramps', 'Sprain', 
                             'Indigestion', 'Toothache', 'Answer 1', 'Answer 2']
        
        for col in categorical_columns:
            if col in df_processed.columns:
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                self.label_encoders[col] = le
        
        # Encode target variable
        if self.target_column in df_processed.columns:
            le_target = LabelEncoder()
            df_processed[self.target_column] = le_target.fit_transform(df_processed[self.target_column])
            self.label_encoders[self.target_column] = le_target
        
        # Define feature columns (exclude target and follow-up questions)
        self.feature_columns = ['Gender', 'Age', 'Weight', 'Fever', 'Common Cold', 'Cough', 
                               'Body Pain', 'Headache', 'Menstrual Cramps', 'Sprain', 
                               'Indigestion', 'Toothache', 'Answer 1', 'Answer 2']
        
        # Filter only columns that exist in the dataset
        self.feature_columns = [col for col in self.feature_columns if col in df_processed.columns]
        
        logger.info(f"Feature columns: {self.feature_columns}")
        logger.info(f"Target column: {self.target_column}")
        
        self.df_processed = df_processed
        return True
    
    def train_model(self):
        """Train the Random Forest ensemble classifier"""
        if not hasattr(self, 'df_processed'):
            logger.error("Data not preprocessed. Call preprocess_data() first.")
            return False
        
        # Prepare features and target
        X = self.df_processed[self.feature_columns]
        y = self.df_processed[self.target_column]
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Model trained successfully. Accuracy: {accuracy:.4f}")
        logger.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")
        
        return True
    
    def save_model(self, model_path="ensemble_model.pkl"):
        """Save the trained model and encoders"""
        if self.model is None:
            logger.error("Model not trained. Call train_model() first.")
            return False
        
        model_data = {
            'model': self.model,
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {model_path}")
        return True
    
    def load_model(self, model_path="ensemble_model.pkl"):
        """Load the trained model and encoders"""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.label_encoders = model_data['label_encoders']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            self.target_column = model_data['target_column']
            
            logger.info("Model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def predict(self, features_dict):
        """Predict recommendation based on features"""
        if self.model is None:
            logger.error("Model not loaded. Call load_model() first.")
            return None
        
        # Prepare features
        features = []
        for col in self.feature_columns:
            if col in features_dict:
                value = features_dict[col]
                # Encode if encoder exists
                if col in self.label_encoders:
                    value = self.label_encoders[col].transform([str(value)])[0]
                features.append(value)
            else:
                # Default value if feature not provided
                features.append(0)
        
        # Scale features
        features_scaled = self.scaler.transform([features])
        
        # Predict
        prediction = self.model.predict(features_scaled)[0]
        
        # Decode prediction
        if self.target_column in self.label_encoders:
            prediction_decoded = self.label_encoders[self.target_column].inverse_transform([prediction])[0]
        else:
            prediction_decoded = prediction
        
        return prediction_decoded
    
    def get_follow_up_questions(self, symptoms, age, weight, gender):
        """Get follow-up questions based on symptoms and demographics"""
        if self.df is None:
            return []
        
        # Filter dataset based on symptoms and demographics
        mask = pd.Series([True] * len(self.df))
        
        # Filter by demographics
        if age:
            mask &= (self.df['Age'] == age)
        if weight:
            mask &= (self.df['Weight'] == weight)
        if gender:
            mask &= (self.df['Gender'] == gender)
        
        # Filter by symptoms
        for symptom in symptoms:
            if symptom in self.df.columns:
                mask &= (self.df[symptom] == 'Yes')
        
        filtered_df = self.df[mask]
        
        if filtered_df.empty:
            return []
        
        # Get follow-up questions from the first matching row
        row = filtered_df.iloc[0]
        questions = []
        
        for i in range(1, 5):
            question_col = f'Follow up question {i}'
            if question_col in row and pd.notna(row[question_col]) and row[question_col] != '':
                questions.append({
                    'question': row[question_col],
                    'question_number': i
                })
        
        return questions

if __name__ == "__main__":
    # Test the dataset processor
    processor = DatasetProcessor()
    
    if processor.load_dataset():
        if processor.preprocess_data():
            if processor.train_model():
                processor.save_model()
                print("Model training completed successfully!")
            else:
                print("Model training failed!")
        else:
            print("Data preprocessing failed!")
    else:
        print("Dataset loading failed!") 