import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import re
import json
from typing import List, Dict, Set, Optional
from collections import Counter

EXCEL_PATH = r'C:\Users\Supriya S\OneDrive\Desktop\IDP\Data from books .xlsx'

# ---- Symptom flag mapping and binning helpers ----

SYMPTOM_FLAGS = {
    'fever': ['fever', 'temperature', 'high temp', 'pyrexia'],
    'common cold': ['common cold', 'cold', 'runny nose', 'sneezing'],
    'cough': ['cough', 'coughing'],
    'body pain': ['body pain', 'body ache', 'muscle pain', 'aches'],
    'headache': ['headache', 'migraine', 'head pain'],
    'menstrual cramps': ['menstrual cramps', 'period pain', 'cramps'],
    'sprain': ['sprain', 'twisted ankle', 'twisted wrist', 'muscle sprain'],
    'indigestion': ['indigestion', 'stomach ache', 'tummy ache', 'upset stomach'],
    'toothache': ['toothache', 'tooth pain', 'dental pain']
}

AGE_BINS = [(18, 25), (26, 35), (36, 50), (51, 80)]
WEIGHT_BINS = [(40, 60), (61, 90), (91, 300)]  # 300 as upper bound


def bin_value(value, bins):
    for b in bins:
        if b[0] <= value <= b[1]:
            return f"{b[0]}–{b[1]}"
    return None

def map_symptom_to_flag(symptom):
    s = symptom.lower().strip()
    for flag, synonyms in SYMPTOM_FLAGS.items():
        if any(word in s for word in synonyms):
            return flag
    return None

# Helper functions to map numeric input to string bucket

def age_to_bucket(age, age_buckets):
    for bucket in age_buckets:
        # bucket is like '18-25', '26-35', etc.
        try:
            low, high = map(int, bucket.split('-'))
            if low <= age <= high:
                return bucket
        except Exception:
            continue
    return None

def weight_to_bucket(weight, weight_buckets):
    for bucket in weight_buckets:
        try:
            low, high = map(int, bucket.replace('+','-999').split('-'))
            if low <= weight <= high:
                return bucket
        except Exception:
            continue
    return None

class MedicalProfileMatcher:
    def __init__(self, excel_path='new_dataset.xlsx'):
        self.df = pd.read_excel(excel_path)
        # Normalize all columns to Title Case for robust lookups
        self.df.columns = [c.strip().title() for c in self.df.columns]
        # Age and Weight are already string buckets
        self.df['AgeBin'] = self.df['Age']
        self.df['WeightBin'] = self.df['Weight']
        self.age_buckets = sorted(self.df['AgeBin'].dropna().unique(), key=lambda x: int(x.split('-')[0]))
        self.weight_buckets = ["40-60", "60-90", "90+"]
        self.df['Gender'] = self.df['Gender'].str.lower()
        for flag in SYMPTOM_FLAGS:
            flag_title = flag.title()
            if flag_title in self.df.columns:
                self.df[flag_title] = self.df[flag_title].astype(str).str.lower()

    def match_row(self, gender, age, weight, yes_symptoms: list):
        # Map numeric age/weight to string bucket
        age_bin = age_to_bucket(age, self.age_buckets)
        weight_bin = weight_to_bucket(weight, self.weight_buckets)
        gender = gender.lower().strip()
        if not age_bin or not weight_bin:
            return None
        df = self.df[
            (self.df['Gender'] == gender) &
            (self.df['AgeBin'] == age_bin) &
            (self.df['WeightBin'] == weight_bin)
        ]
        for flag in yes_symptoms:
            flag_title = flag.title()
            if flag_title in df.columns:
                df = df[df[flag_title] == 'yes']
        if df.empty:
            return None
        row = df.iloc[0]
        def safe_get(col):
            val = row.get(col, '')
            if pd.isna(val) or str(val).strip().lower() == 'nan':
                return ''
            return str(val).strip()
        followups = {
            'Follow up question 1': safe_get('Follow Up Question 1'),
            'Answer 1': safe_get('Answer 1'),
            'Follow up question 2': safe_get('Follow Up Question 2'),
            'Answer 2': safe_get('Answer 2'),
            'Follow up question 3': safe_get('Follow Up Question 3'),
            'Follow up question 4': safe_get('Follow Up Question 4'),
            'OTC/Doc': safe_get('Otc/Doc')
        }
        return followups

class DataLoader:
    def __init__(self, file_path=r'C:\Users\Supriya S\OneDrive\Desktop\IDP\Data from books_updated.xlsx'):
        self.df = pd.read_excel(file_path)
        self.medicines = []
        self.symptom_relations = {}
        self._prepare_data()

    def _prepare_data(self):
        for _, row in self.df.iterrows():
            # Parse min_age and max_age from Age Restrictions
            age_restr = str(row['Age Restrictions'])
            if '-' in age_restr:
                min_age, max_age = map(int, age_restr.split('-'))
            else:
                min_age, max_age = 0, 100  # fallback/default

            medicine = {
                'medicine': row['Drug Name(s)'],
                'symptom_category': row['Symptom Category'],
                'min_age': min_age,
                'max_age': max_age,
                'gender': row['Gender'],
                'symptoms': row['Symptom (Patient-Friendly)'],
                'dosage': row.get('Dosage', ''),
                'frequency': row.get('Frequency', ''),
                'duration': row.get('Duration', '')
            }
            self.medicines.append(medicine)

            # Build symptom relations for follow-up logic
            symptoms = [s.strip().lower() for s in str(row['Symptom (Patient-Friendly)']).split(',')]
            for i, symptom in enumerate(symptoms):
                if symptom not in self.symptom_relations:
                    self.symptom_relations[symptom] = set()
                self.symptom_relations[symptom].update(symptoms[i+1:])

    def find_matching_medicines(self, symptoms, age, gender):
        symptoms = [s.lower().strip() for s in symptoms]
        matches = []
        for med in self.medicines:
            med_symptoms = [s.strip().lower() for s in str(med['symptoms']).split(',')]
            if any(s in med_symptoms for s in symptoms):
                if med['min_age'] <= age <= med['max_age']:
                    if str(med['gender']).lower() == 'all' or str(med['gender']).lower() == str(gender).lower():
                        matches.append(med)
        return matches

    def get_related_symptoms(self, symptom):
        return list(self.symptom_relations.get(symptom.lower(), set()))

class MedicalDataLoader:
    def __init__(self):
        self.data = None
        self.symptom_relations = {}
        self.symptom_categories = {}
        self.load_data()

    def _clean_symptom_name(self, symptom_str: str) -> str:
        """Cleans a symptom string by removing redundant words and normalizing it."""
        if not isinstance(symptom_str, str):
            return ""
        cleaned = symptom_str.lower().strip()
        # Remove common redundant phrases and words that might be part of questions
        cleaned = cleaned.replace(" only", "").replace(" just", "").replace(" and", "").replace(" or", "").replace(" also", "").replace(" with", "").strip()
        # Replace commas with spaces, then re-strip to handle cases like 'symptom1,symptom2'
        cleaned = cleaned.replace(",", " ").strip()
        # Remove any extra spaces
        cleaned = ' '.join(cleaned.split())
        return cleaned

    def load_data(self):
        """Load and prepare the medical dataset"""
        try:
            self.data = pd.read_excel("updated_dataset.xlsx")
            self._prepare_data()
        except Exception as e:
            print(f"Error loading data: {e}")
            self.data = pd.DataFrame()

    def _prepare_data(self):
        """Prepare the data for symptom matching and medicine recommendations"""
        # Convert standardized_symptoms to string and lowercase
        self.data['standardized_symptoms'] = self.data['standardized_symptoms'].astype(str).str.lower()
        
        # Build symptom relations and categories
        for _, row in self.data.iterrows():
            if pd.notna(row['standardized_symptoms']):
                # Split by semicolon and comma, then clean each individual symptom
                raw_symptom_parts = []
                for part in str(row['standardized_symptoms']).split(';'):
                    raw_symptom_parts.extend(part.split(','))
                
                symptoms = [self._clean_symptom_name(s) for s in raw_symptom_parts if self._clean_symptom_name(s)]
                symptoms = sorted(list(set(symptoms))) # Remove duplicates and sort for consistent sets

                category = row['Symptom Category']
                
                # Add to symptom categories
                for symptom in symptoms:
                    self.symptom_categories[symptom] = category
                
                # Build relations between symptoms
                for i, symptom1 in enumerate(symptoms):
                    for symptom2 in symptoms[i+1:]:
                        if symptom1 not in self.symptom_relations:
                            self.symptom_relations[symptom1] = set()
                        if symptom2 not in self.symptom_relations:
                            self.symptom_relations[symptom2] = set()
                        self.symptom_relations[symptom1].add(symptom2)
                        self.symptom_relations[symptom2].add(symptom1)

    def get_symptom_category(self, symptom: str) -> str:
        """Get the category for a given symptom"""
        symptom = self._clean_symptom_name(symptom)
        return self.symptom_categories.get(symptom)

    def get_related_symptoms(self, symptom: str) -> List[str]:
        """Get symptoms that commonly occur with the given symptom"""
        symptom = self._clean_symptom_name(symptom)
        
        # Get directly related symptoms from co-occurrence in dataset rows
        directly_related = list(self.symptom_relations.get(symptom, set()))
        
        # Filter out the primary symptom itself from directly_related
        directly_related = [s for s in directly_related if s != symptom]

        if directly_related:
            return directly_related

        # If no directly related symptoms, find all symptoms in the same category
        category = self.get_symptom_category(symptom)
        if category:
            category_symptoms = set()
            for s, cat in self.symptom_categories.items():
                cleaned_s = self._clean_symptom_name(s)
                if cat == category and cleaned_s != symptom and cleaned_s:
                    category_symptoms.add(cleaned_s)
            return list(category_symptoms)
        
        return []

    def find_matching_medicines(self, symptoms: List[str], age: int, gender: str) -> Optional[Dict]:
        """Find matching medicines based on symptoms, age, and gender"""
        if self.data.empty:
            return None

        # Clean and convert user symptoms to a set for efficient checking
        user_symptoms_cleaned = {self._clean_symptom_name(s) for s in symptoms if self._clean_symptom_name(s)}
        
        # Find matching rows
        matches = []
        for _, row in self.data.iterrows():
            if pd.notna(row['standardized_symptoms']):
                # Clean and parse row symptoms
                raw_row_symptom_parts = []
                for part in str(row['standardized_symptoms']).split(';'):
                    raw_row_symptom_parts.extend(part.split(','))
                
                row_symptoms_cleaned = {self._clean_symptom_name(s) for s in raw_row_symptom_parts if self._clean_symptom_name(s)}
                
                # Check if all cleaned user symptoms are a subset of the row's cleaned symptoms
                if user_symptoms_cleaned.issubset(row_symptoms_cleaned):
                    # Check age restrictions
                    age_restrictions = str(row['Age Restrictions']).lower()
                    if age_restrictions != 'nan' and age_restrictions:
                        if 'under' in age_restrictions:
                            max_age = int(''.join(filter(str.isdigit, age_restrictions)))
                            if age >= max_age:
                                continue
                        elif 'over' in age_restrictions:
                            min_age = int(''.join(filter(str.isdigit, age_restrictions)))
                            if age < min_age:
                                continue
                    
                    # Check gender restrictions
                    if str(row['GENDER']).lower() != 'any' and str(row['GENDER']).lower() != gender.lower():
                        continue
                    
                    # Only include OTC medicines
                    if row['is_otc']:
                        matches.append({
                            'drug_name': row['Drug Name(s)'],
                            'drug_class': row['Drug Class'],
                            'route': row['Route'],
                            'otc_prescription': row['OTC/Prescription'],
                            'contraindications': row['Contraindications / Cautions'],
                            'recommendation': row['Recommendation']
                        })
        
        # If multiple matches with the same set of symptoms and same medicine, return the first.
        if matches:
            # For now, just return the first match after ensuring cleaned symptoms are used.
            return matches[0] if matches else None
        
        return None

def load_medical_data():
    raw_df = pd.read_excel(EXCEL_PATH)
    print("Original columns from new dataset:", raw_df.columns.tolist())

    # Normalize column names for easier access
    col_map = {
        'Symptom (Patient-Friendly)': 'SYMPTOM',
        'Symptom Category': 'SYMPTOM_CATEGORY',
        'Age Restrictions': 'AGE_RESTRICTIONS',
        'Gender': 'GENDER'
    }
    df = raw_df.rename(columns=col_map)

    # Select relevant columns for chatbot logic
    # We are interested in Symptom, Symptom Category, Age Restrictions, Gender
    # We also need to extract individual symptoms from the 'SYMPTOM' column if they are comma-separated
    required_columns = ['SYMPTOM', 'SYMPTOM_CATEGORY', 'AGE_RESTRICTIONS', 'GENDER']
    df = df[required_columns].copy()

    # Clean and preprocess symptom data
    df['SYMPTOM'] = df['SYMPTOM'].astype(str).str.lower().str.strip()
    df['SYMPTOM_CATEGORY'] = df['SYMPTOM_CATEGORY'].astype(str).str.lower().str.strip()
    df['AGE_RESTRICTIONS'] = df['AGE_RESTRICTIONS'].astype(str).str.lower().str.strip()
    df['GENDER'] = df['GENDER'].astype(str).str.lower().str.strip()

    return df

if __name__ == "__main__":
    df = load_medical_data()
    print("Processed data head:")
    print(df.head())
    print("Unique Symptom Categories:", df['SYMPTOM_CATEGORY'].unique().tolist())
    print("Unique Symptoms:", df['SYMPTOM'].unique().tolist())


    def analyze_symptom_correlations(self, primary_symptom: str, threshold: float = 0.3) -> Dict[str, float]:
        """Analyze how frequently other symptoms co-occur with the primary symptom"""
        primary_col = primary_symptom.title()
        if primary_col not in self.df.columns:
            return {}
            
        # Filter rows where primary symptom is 'yes'
        filtered_df = self.df[self.df[primary_col].str.lower() == 'yes']
        total_rows = len(filtered_df)
        
        if total_rows == 0:
            return {}
            
        correlations = {}
        # Check other symptoms
        for symptom in SYMPTOM_FLAGS.keys():
            symptom_col = symptom.title()
            if symptom_col == primary_col or symptom_col not in self.df.columns:
                continue
                
            # Calculate co-occurrence frequency
            yes_count = len(filtered_df[filtered_df[symptom_col].str.lower() == 'yes'])
            correlation = yes_count / total_rows
            
            if correlation >= threshold:
                correlations[symptom] = correlation
                
        return dict(sorted(correlations.items(), key=lambda x: x[1], reverse=True))
        
    def get_cooccurring_symptoms(self, primary_symptom: str, min_frequency: float = 0.3) -> List[str]:
        """Find symptoms that frequently co-occur with the primary symptom"""
        primary_col = primary_symptom.title()
        if primary_col not in self.df.columns:
            return []
            
        # Filter rows where primary symptom is 'yes'
        filtered_df = self.df[self.df[primary_col].str.lower() == 'yes']
        total_rows = len(filtered_df)
        
        if total_rows == 0:
            return []
            
        cooccurring = []
        for symptom in SYMPTOM_FLAGS.keys():
            symptom_col = symptom.title()
            if symptom_col == primary_col or symptom_col not in self.df.columns:
                continue
                
            # Count how often this symptom is 'yes' in the filtered rows
            yes_count = len(filtered_df[filtered_df[symptom_col].str.lower() == 'yes'])
            frequency = yes_count / total_rows
            
            if frequency >= min_frequency:
                cooccurring.append(symptom)
                
        return cooccurring
    
    def get_followup_questions(self, symptoms: List[str]) -> List[Dict]:
        """Get follow-up questions from rows matching the symptoms"""
        # Filter rows where any of the symptoms is 'yes'
        mask = pd.Series(False, index=self.df.index)
        for symptom in symptoms:
            symptom_col = symptom.title()
            if symptom_col in self.df.columns:
                mask |= (self.df[symptom_col].str.lower() == 'yes')
        
        filtered_df = self.df[mask]
        if len(filtered_df) == 0:
            return []
        
        followup_questions = []
        # Get questions 1-3 if they exist and are not empty
        for i in range(1, 4):
            q_col = f'Follow up question {i}'
            a_col = f'Answer {i}'
            
            if q_col in filtered_df.columns:
                questions = filtered_df[[q_col, a_col]].dropna(subset=[q_col]).drop_duplicates()
                for _, row in questions.iterrows():
                    q = str(row[q_col]).strip()
                    a = str(row[a_col]).strip() if pd.notna(row[a_col]) else ''
                    if q:
                        followup_questions.append({
                            'question': q,
                            'answer': a,
                            'question_number': i
                        })
        
        # Always add question 4 if it exists
        q4_col = 'Follow up question 4'
        if q4_col in filtered_df.columns:
            q4s = filtered_df[q4_col].dropna().unique()
            for q in q4s:
                q = str(q).strip()
                if q:
                    followup_questions.append({
                        'question': q,
                        'question_number': 4,
                        'answer': None  # No answer needed for question 4
                    })
        
        return followup_questions