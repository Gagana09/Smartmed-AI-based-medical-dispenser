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
    """
    def __init__(self):
        self.model = None
        self.model_type = None
        self.symptom_columns = SYMPTOMS
        self.trained = False
        self.feature_importances_ = None
        self.association_rules_ = None
        self.benchmark_scores_ = {}

    def train_on_csv(self, csv_path: str):
        """
        Trains the ML model(s) on the given CSV file for symptom co-occurrence analysis.
        Benchmarks Association Rules, Logistic Regression, Decision Tree, and Naive Bayes.
        Selects the best model based on mean ROC-AUC or F1 score.
        """
        df = pd.read_csv(csv_path) if csv_path.endswith('.csv') else pd.read_excel(csv_path, engine="openpyxl")
        # Only use the 9 primary symptom columns for ML
        X = df[self.symptom_columns].fillna("No").apply(lambda col: col.map(lambda x: str(x).strip().lower() == "yes"))
        X = X.astype(int)
        y = X.copy()  # Multi-label: each symptom is a target

        # 1. Association Rules (mlxtend)
        # Prepare transactions for apriori
        transactions = X.apply(lambda row: [col for col in X.columns if row[col] == 1], axis=1).tolist()
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df_apriori = pd.DataFrame(te_ary, columns=te.columns_)
        frequent_itemsets = apriori(df_apriori, min_support=0.01, use_colnames=True)
        assoc_rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)
        self.association_rules_ = assoc_rules
        # Association rules don't have ROC-AUC, so we use coverage of rules as a proxy
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
            # Feature importances: mean absolute value of coefficients
            self.feature_importances_ = np.mean(np.abs(lr.estimators_[0].coef_), axis=0)
        elif best_model_type == "decision_tree":
            self.model = dt
            # Feature importances: mean of feature_importances_ across estimators
            self.feature_importances_ = np.mean([est.feature_importances_ for est in dt.estimators_], axis=0)
        elif best_model_type == "naive_bayes":
            self.model = nb
            # Feature importances: mean of class_log_prior_ (not very interpretable)
            self.feature_importances_ = np.mean([est.class_log_prior_ for est in nb.estimators_], axis=0)
        else:
            self.model = None  # Association rules are not a scikit-learn model
            self.feature_importances_ = None
        self.trained = True

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