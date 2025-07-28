import pandas as pd
import numpy as np
from typing import List, Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score, f1_score
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import warnings
import json
import os
from datetime import datetime
warnings.filterwarnings('ignore')

SYMPTOMS = [
    "Fever", "Common Cold", "Cough", "Body Pain", "Headache",
    "Menstrual Cramps", "Sprain", "Indigestion", "Toothache"
]

class SymptomPredictor:
    """
    ML-based predictor for symptom co-occurrence and ranking.
    Trains on a CSV with symptom columns and provides methods to predict next best symptom to ask.
    Benchmarks multiple models and uses the best one for predictions.
    Now includes real-time learning from user interactions.
    """
    def __init__(self, real_time_learning=True):
        self.model = None
        self.model_type = None
        self.symptom_columns = SYMPTOMS
        self.trained = False
        self.feature_importances_ = None
        self.association_rules_ = None
        self.benchmark_scores_ = {}
        
        # Real-time learning components
        self.real_time_learning = real_time_learning
        self.user_interactions = []
        self.interaction_file = "user_interactions.json"
        self.min_interactions_for_retrain = 2  # Retrain every 2 user interactions
        self.last_retrain_count = 0
        
        # Load existing interactions if available
        self._load_user_interactions()

    def _load_user_interactions(self):
        """Load existing user interactions from file"""
        if os.path.exists(self.interaction_file):
            try:
                with open(self.interaction_file, 'r') as f:
                    self.user_interactions = json.load(f)
                print(f"Loaded {len(self.user_interactions)} existing user interactions")
            except Exception as e:
                print(f"Error loading user interactions: {e}")
                self.user_interactions = []

    def _save_user_interactions(self):
        """Save user interactions to file"""
        try:
            with open(self.interaction_file, 'w') as f:
                json.dump(self.user_interactions, f, indent=2)
        except Exception as e:
            print(f"Error saving user interactions: {e}")

    def add_user_interaction(self, demographics: Dict[str, str], symptoms: Dict[str, str], 
                           final_recommendation: str = None, success_rating: int = None):
        """
        Add a new user interaction for real-time learning.
        
        Args:
            demographics: User demographics (age, gender, weight)
            symptoms: Final symptom profile (symptom: Yes/No)
            final_recommendation: What was recommended
            success_rating: User rating 1-5 (optional)
        """
        if not self.real_time_learning:
            return
            
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'demographics': demographics,
            'symptoms': symptoms,
            'recommendation': final_recommendation,
            'success_rating': success_rating
        }
        
        self.user_interactions.append(interaction)
        self._save_user_interactions()
        
        # Check if we should retrain
        if len(self.user_interactions) >= self.min_interactions_for_retrain and \
           len(self.user_interactions) > self.last_retrain_count:
            self._retrain_with_user_data()

    def _retrain_with_user_data(self):
        """Retrain models with accumulated user interaction data"""
        if len(self.user_interactions) < self.min_interactions_for_retrain:
            return
            
        print(f"Retraining models with {len(self.user_interactions)} user interactions...")
        
        # Convert user interactions to training data
        new_data = []
        for interaction in self.user_interactions:
            row = {}
            # Add demographics
            row.update(interaction['demographics'])
            # Add symptoms
            row.update(interaction['symptoms'])
            new_data.append(row)
        
        if len(new_data) < 2:  # Need minimum 2 interactions for meaningful training
            return
            
        # Create temporary CSV for retraining
        temp_df = pd.DataFrame(new_data)
        temp_file = "temp_user_data.csv"
        temp_df.to_csv(temp_file, index=False)
        
        try:
            # Retrain with combined data (original + user data)
            self._retrain_models(temp_file)
            self.last_retrain_count = len(self.user_interactions)
            print("Real-time retraining completed successfully!")
        except Exception as e:
            print(f"Error during real-time retraining: {e}")
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def _retrain_models(self, additional_data_path: str):
        """Retrain models with additional user data"""
        # Load original data
        original_df = pd.read_csv(self.original_data_path) if hasattr(self, 'original_data_path') else None
        
        # Load new user data
        user_df = pd.read_csv(additional_data_path)
        
        # Combine datasets
        if original_df is not None:
            combined_df = pd.concat([original_df, user_df], ignore_index=True)
        else:
            combined_df = user_df
        
        # Retrain using the combined data
        self._train_on_dataframe(combined_df)

    def _train_on_dataframe(self, df: pd.DataFrame):
        """Internal method to train on a DataFrame"""
        # Only use the 9 primary symptom columns for ML
        X = df[self.symptom_columns].fillna("No").apply(lambda col: col.map(lambda x: str(x).strip().lower() == "yes"))
        X = X.astype(int)
        y = X.copy()  # Multi-label: each symptom is a target

        # 1. Association Rules (mlxtend)
        transactions = X.apply(lambda row: [col for col in X.columns if row[col] == 1], axis=1).tolist()
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df_apriori = pd.DataFrame(te_ary, columns=te.columns_)
        frequent_itemsets = apriori(df_apriori, min_support=0.01, use_colnames=True)
        assoc_rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)
        self.association_rules_ = assoc_rules
        assoc_score = assoc_rules.shape[0]
        self.benchmark_scores_["association_rules"] = assoc_score

        # 2. Logistic Regression (OneVsRest)
        lr = OneVsRestClassifier(LogisticRegression(solver='liblinear'))
        lr.fit(X, y)
        y_pred_lr = lr.predict(X)
        try:
            auc_lr = roc_auc_score(y, y_pred_lr, average='macro')
        except Exception:
            auc_lr = f1_score(y, y_pred_lr, average='macro')
        self.benchmark_scores_["logistic_regression"] = auc_lr

        # 3. Decision Tree (OneVsRest)
        dt = OneVsRestClassifier(DecisionTreeClassifier(max_depth=5, random_state=42))
        dt.fit(X, y)
        y_pred_dt = dt.predict(X)
        try:
            auc_dt = roc_auc_score(y, y_pred_dt, average='macro')
        except Exception:
            auc_dt = f1_score(y, y_pred_dt, average='macro')
        self.benchmark_scores_["decision_tree"] = auc_dt

        # 4. Naive Bayes (BernoulliNB)
        nb = OneVsRestClassifier(BernoulliNB())
        nb.fit(X, y)
        y_pred_nb = nb.predict(X)
        try:
            auc_nb = roc_auc_score(y, y_pred_nb, average='macro')
        except Exception:
            auc_nb = f1_score(y, y_pred_nb, average='macro')
        self.benchmark_scores_["naive_bayes"] = auc_nb

        # Pick the best model
        best_model_type = max(self.benchmark_scores_, key=self.benchmark_scores_.get)
        self.model_type = best_model_type
        
        if best_model_type == "logistic_regression":
            self.model = lr
            self.feature_importances_ = np.mean(np.abs(lr.estimators_[0].coef_), axis=0)
        elif best_model_type == "decision_tree":
            self.model = dt
            self.feature_importances_ = np.mean([est.feature_importances_ for est in dt.estimators_], axis=0)
        elif best_model_type == "naive_bayes":
            self.model = nb
            self.feature_importances_ = np.mean([est.class_log_prior_ for est in nb.estimators_], axis=0)
        else:
            self.model = None
            self.feature_importances_ = None
            
        self.trained = True

    def train_on_csv(self, csv_path: str):
        """
        Trains the ML model(s) on the given CSV file for symptom co-occurrence analysis.
        Benchmarks Association Rules, Logistic Regression, Decision Tree, and Naive Bayes.
        Selects the best model based on mean ROC-AUC or F1 score.
        """
        self.original_data_path = csv_path  # Store for retraining
        df = pd.read_csv(csv_path) if csv_path.endswith('.csv') else pd.read_excel(csv_path, engine="openpyxl")
        self._train_on_dataframe(df)

    def get_learning_stats(self) -> Dict[str, Any]:
        """Get statistics about the real-time learning system"""
        return {
            'total_interactions': len(self.user_interactions),
            'real_time_learning_enabled': self.real_time_learning,
            'min_interactions_for_retrain': self.min_interactions_for_retrain,
            'last_retrain_count': self.last_retrain_count,
            'model_type': self.model_type,
            'benchmark_scores': self.benchmark_scores_,
            'trained': self.trained
        }

    def predict_symptom_probabilities(self, current_symptoms: Dict[str, str], demographics: Dict[str, str]) -> Dict[str, float]:
        """
        Given current confirmed symptoms and demographics, predicts the probability of each remaining symptom being 'Yes'.
        Returns a dict of symptom: probability.
        """
        if not self.trained:
            raise Exception("Model not trained. Call train_on_csv first.")
        # Prepare input vector
        x = np.array([[current_symptoms.get(s, "No") == "Yes" for s in self.symptom_columns]]).astype(int)
        symptom_probs = {}
        if self.model_type in ["logistic_regression", "decision_tree", "naive_bayes"]:
            proba = self.model.predict_proba(x)
            for idx, s in enumerate(self.symptom_columns):
                # proba is a list of arrays, one per symptom
                # proba[idx][0][1] is the probability of 'Yes' for symptom idx
                if isinstance(proba[idx], np.ndarray):
                    symptom_probs[s] = proba[idx][0][1]
                else:
                    symptom_probs[s] = proba[0][idx][1]
        elif self.model_type == "association_rules":
            # Use association rules to estimate probabilities
            # For each unconfirmed symptom, find rules where antecedents are a subset of current 'Yes' symptoms
            yes_symptoms = [s for s, v in current_symptoms.items() if v == "Yes"]
            for s in self.symptom_columns:
                if s in yes_symptoms:
                    symptom_probs[s] = 1.0
                    continue
                # Find rules where antecedents are subset of yes_symptoms and consequent is s
                rules = self.association_rules_[
                    (self.association_rules_['consequents'].apply(lambda x: s in x)) &
                    (self.association_rules_['antecedents'].apply(lambda x: set(x).issubset(set(yes_symptoms))))
                ]
                if not rules.empty:
                    # Use the highest confidence as the probability
                    symptom_probs[s] = rules['confidence'].max()
                else:
                    symptom_probs[s] = 0.0
        else:
            raise Exception("Unknown model type.")
        return symptom_probs

    def get_next_best_symptom(self, current_symptoms: Dict[str, str], available_symptoms: List[str], demographics: Dict[str, str]) -> List[Dict[str, Any]]:
        """
        Ranks available symptoms by predicted probability and returns the top ones with confidence scores.
        Returns a list of dicts: { 'symptom': str, 'probability': float }
        """
        symptom_probs = self.predict_symptom_probabilities(current_symptoms, demographics)
        ranked = [
            {'symptom': s, 'probability': symptom_probs[s]}
            for s in available_symptoms if s in symptom_probs
        ]
        ranked = sorted(ranked, key=lambda d: d['probability'], reverse=True)
        return ranked

    def get_symptom_importance(self) -> Dict[str, float]:
        """
        Returns feature importance or association strengths for each symptom (for debugging/interpretability).
        """
        if self.model_type in ["logistic_regression", "decision_tree", "naive_bayes"] and self.feature_importances_ is not None:
            return {s: float(imp) for s, imp in zip(self.symptom_columns, self.feature_importances_)}
        elif self.model_type == "association_rules":
            # For association rules, return the mean confidence for each symptom as consequent
            import collections
            confs = collections.defaultdict(list)
            for _, row in self.association_rules_.iterrows():
                for s in row['consequents']:
                    confs[s].append(row['confidence'])
            return {s: float(np.mean(confs[s])) if confs[s] else 0.0 for s in self.symptom_columns}
        else:
            return {s: 0.0 for s in self.symptom_columns} 