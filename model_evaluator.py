import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import warnings
warnings.filterwarnings('ignore')
#symptom list
SYMPTOMS = ["Fever", "Common Cold", "Cough", "Body Pain", "Headache",
            "Menstrual Cramps", "Sprain", "Indigestion", "Toothache"]

class ModelEvaluator:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.symptom_columns = SYMPTOMS
        self.models = {}
        self.results = {}
        
    def load_data(self):
        """Load and prepare dataset"""
        print(f"Loading dataset: {self.dataset_path}")
        df = pd.read_excel(self.dataset_path, engine="openpyxl")
        
        X = df[self.symptom_columns].fillna("No").apply(
            lambda col: col.map(lambda x: str(x).strip().lower() == "yes")
        ).astype(int)
        
        y = X.copy()
        return X.values, y.values
    
    def train_models(self, X, y):
        """Train all models"""
        print("Training models...")
        
        # Logistic Regression
        lr = OneVsRestClassifier(LogisticRegression(solver='liblinear', random_state=42))
        lr.fit(X, y)
        self.models['logistic_regression'] = lr
        
        # Decision Tree
        dt = OneVsRestClassifier(DecisionTreeClassifier(max_depth=5, random_state=42))
        dt.fit(X, y)
        self.models['decision_tree'] = dt
        
        # Naive Bayes
        nb = OneVsRestClassifier(BernoulliNB())
        nb.fit(X, y)
        self.models['naive_bayes'] = nb
        
        # Association Rules
        transactions = pd.DataFrame(X, columns=self.symptom_columns).apply(
            lambda row: [col for col in self.symptom_columns if row[col] == 1], axis=1
        ).tolist()
        
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df_apriori = pd.DataFrame(te_ary, columns=te.columns_)
        frequent_itemsets = apriori(df_apriori, min_support=0.01, use_colnames=True)
        assoc_rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)
        self.models['association_rules'] = assoc_rules
        
        print("All models trained!")
    
    def evaluate_model(self, model_name, model, X, y):
        """Evaluate a single model"""
        print(f"Evaluating {model_name}...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        if model_name != 'association_rules':
            # Retrain on split
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            # Calculate metrics
            f1 = f1_score(y_test, y_pred, average='macro')
            acc = accuracy_score(y_test, y_pred)
            try:
                auc = roc_auc_score(y_test, y_pred_proba, average='macro')
            except:
                auc = 0.0
            
            # Cross-validation
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='f1_macro')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
        else:
            # Association rules evaluation
            y_pred = self._predict_assoc_rules(X_test, model)
            f1 = f1_score(y_test, y_pred, average='macro')
            acc = accuracy_score(y_test, y_pred)
            auc = 0.0  # Not applicable for association rules
            cv_mean = 0.0
            cv_std = 0.0
        
        return {
            'f1_score': f1,
            'accuracy': acc,
            'roc_auc': auc,
            'cv_f1_mean': cv_mean,
            'cv_f1_std': cv_std
        }
    
    def _predict_assoc_rules(self, X_test, rules):
        """Predict using association rules"""
        predictions = np.zeros_like(X_test)
        
        for i, row in enumerate(X_test):
            present_symptoms = [self.symptom_columns[j] for j, val in enumerate(row) if val == 1]
            
            for j, symptom in enumerate(self.symptom_columns):
                if symptom in present_symptoms:
                    predictions[i, j] = 1
                    continue
                
                relevant_rules = rules[
                    (rules['consequents'].apply(lambda x: symptom in x)) &
                    (rules['antecedents'].apply(lambda x: set(x).issubset(set(present_symptoms))))
                ]
                
                if not relevant_rules.empty:
                    max_confidence = relevant_rules['confidence'].max()
                    if max_confidence > 0.5:
                        predictions[i, j] = 1
        
        return predictions
    
    def evaluate_all_models(self):
        """Evaluate all models"""
        print("=== MODEL EVALUATION ===")
        
        # Load data
        X, y = self.load_data()
        print(f"Dataset shape: {X.shape}")
        
        # Train models
        self.train_models(X, y)
        
        # Evaluate each model
        for model_name, model in self.models.items():
            self.results[model_name] = self.evaluate_model(model_name, model, X, y)
        
        # Find best model
        best_model = max(self.results.keys(), key=lambda k: self.results[k]['f1_score'])
        
        # Print results
        self._print_results(best_model)
    
    def _print_results(self, best_model):
        """Print evaluation results without CV F1"""
        print("\n" + "="*60)
        print("MODEL EVALUATION RESULTS")
        print("="*60)
        
        print(f"\n{'Model':<20} {'F1 Score':<12} {'Accuracy':<12} {'ROC-AUC':<12}")
        print("-" * 60)
        
        for model_name, results in self.results.items():
            f1 = results['f1_score']
            acc = results['accuracy']
            auc = results['roc_auc']
            
            print(f"{model_name:<20} {f1:<12.4f} {acc:<12.4f} {auc:<12.4f}")
        
        print(f"\nBEST MODEL: {best_model}")
        print(f"Best F1 Score: {self.results[best_model]['f1_score']:.4f}")
        print(f"Best Accuracy: {self.results[best_model]['accuracy']:.4f}")
        print(f"Best ROC-AUC: {self.results[best_model]['roc_auc']:.4f}")

def main():
    """Run the evaluation"""
    evaluator = ModelEvaluator('dataset_1.xlsx')
    evaluator.evaluate_all_models()

if __name__ == "__main__":
    main()
