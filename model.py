# SmartMedAI: Predict Tablet, Dosage, Frequency & Duration
# Reads Excel with possibly misnamed columns and normalizes headers

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import joblib
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import re
import os

# 2. Load & Normalize Columns
EXCEL_PATH = r'C:\Users\Supriya S\OneDrive\Desktop\IDP\Data from books .xlsx'  # updated path
raw = pd.read_excel(EXCEL_PATH)
print("Original columns:", raw.columns.tolist())

# New columns in the dataset:
# Gender, Symptom Category, Symptom (Patient-Friendly), Severity, Drug Class, Drug Name(s), Route, OTC/Prescription, Age Restrictions, Contraindications / Cautions, Follow-up Question(s), Recommendation

# For demonstration, let's just print the first few rows and columns
print(raw.head())
# If you want to do further processing, add it here based on your new requirements.

# Rename known headers:
# If your symptom text is under 'HEALTH RECORDS', rename it to 'SYMPTOMS'
col_map = {
    'HEALTH RECORDS': 'SYMPTOMS',
    'TABLET_PRESCIBED': 'TABLET_PRESCIBED',
    'DOSAGE': 'DOSAGE',
    'FREQUENCY': 'FREQUENCY',
    'DURATION': 'DURATION',
    'AGE': 'AGE',
    'GENDER': 'GENDER'
    # add more mappings if needed
}
df = raw.rename(columns=col_map)

# Drop irrelevant/unnamed columns
keep = list(col_map.values())
df = df[[c for c in df.columns if c in keep]]
print("Normalized columns:", df.columns.tolist())

# 3. Filter & Clean
# Ensure required columns exist
required = ['SYMPTOMS','AGE','TABLET_PRESCIBED','DOSAGE','FREQUENCY','DURATION']
missing = [c for c in required if c not in df.columns]
if missing:
    raise KeyError(f"Missing required columns: {missing}")

# Drop rows with missing symptom or age
df = df[df['SYMPTOMS'].notna() & df['AGE'].notna()].copy()
# Lowercase and strip text columns
df['symptom_text'] = df['SYMPTOMS'].astype(str).str.lower().str.strip()
for tcol in ['TABLET_PRESCIBED','DOSAGE','FREQUENCY','DURATION']:
    df[tcol] = df[tcol].astype(str).str.lower().str.strip()
# Normalize age
df['age_norm'] = df['AGE'] / df['AGE'].max()

# 4. Feature Engineering
vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1,2))
X_text = vectorizer.fit_transform(df['symptom_text'])
X_age  = csr_matrix(df[['age_norm']].values)
X = hstack([X_text, X_age])

# 5. Prepare Multi-Output Targets
y = df[['TABLET_PRESCIBED','DOSAGE','FREQUENCY','DURATION']]

# 6. Train Multi-Output Model
base_clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight='balanced',
    random_state=42
)
model = MultiOutputClassifier(base_clf)
model.fit(X, y)
print(f"Trained on {X.shape[0]} samples; outputs: {y.shape[1]} fields.")

# 7. Save Artifacts
joblib.dump(model, 'med_multioutput.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
print("Saved: med_multioutput.pkl, tfidf_vectorizer.pkl")

class MedicalDataLoader:
    def __init__(self, data_file):
        """Initialize the data loader with the Excel file path"""
        self.data_file = data_file
        self.data = None
        self.symptom_encoder = LabelEncoder()
        self.medicine_encoder = LabelEncoder()
        self.model = None
        self.load_data()
        
    def load_data(self):
        """Load and preprocess the medical data"""
        try:
            # Read the Excel file
            self.data = pd.read_excel(self.data_file)
            
            # Clean and preprocess the data
            self.data['Symptoms'] = self.data['Symptoms'].str.lower()
            self.data['Medicine'] = self.data['Medicine'].str.lower()
            
            # Encode symptoms and medicines
            self.symptom_encoder.fit(self.data['Symptoms'])
            self.medicine_encoder.fit(self.data['Medicine'])
            
            # Create feature matrix
            X = self.symptom_encoder.transform(self.data['Symptoms'])
            y = self.medicine_encoder.transform(self.data['Medicine'])
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X.reshape(-1, 1), y, test_size=0.2, random_state=42
            )
            
            # Train the model
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X_train, y_train)
            
            # Evaluate the model
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Model accuracy: {accuracy:.2f}")
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise
            
    def predict_medicine(self, gender, symptoms, age):
        """Predict medicine based on symptoms, gender, and age"""
        try:
            # Preprocess input
            symptoms = symptoms.lower()
            
            # Encode symptoms
            symptom_encoded = self.symptom_encoder.transform([symptoms])[0]
            
            # Make prediction
            prediction = self.model.predict([[symptom_encoded]])[0]
            
            # Decode prediction
            medicine = self.medicine_encoder.inverse_transform([prediction])[0]
            
            return medicine
            
        except Exception as e:
            print(f"Error predicting medicine: {str(e)}")
            return "Unable to make prediction"
            
    def get_related_symptoms(self, symptom):
        """Get related symptoms based on the input symptom"""
        try:
            # Convert symptom to lowercase
            symptom = symptom.lower()
            
            # Find similar symptoms in the dataset
            similar_symptoms = self.data[
                self.data['Symptoms'].str.contains(symptom, case=False)
            ]['Symptoms'].unique()
            
            # Return up to 5 related symptoms
            return list(similar_symptoms)[:5]
            
        except Exception as e:
            print(f"Error getting related symptoms: {str(e)}")
            return []
            
    def get_symptom_category(self, symptom):
        """Get the category of a symptom"""
        try:
            # Convert symptom to lowercase
            symptom = symptom.lower()
            
            # Find the category in the dataset
            category = self.data[
                self.data['Symptoms'].str.contains(symptom, case=False)
            ]['Category'].iloc[0]
            
            return category
            
        except Exception as e:
            print(f"Error getting symptom category: {str(e)}")
            return "Unknown"