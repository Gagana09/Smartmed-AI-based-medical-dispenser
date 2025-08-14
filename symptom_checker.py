import pandas as pd
from symptom_predictor import SymptomPredictor, SYMPTOMS
from typing import Dict, List, Any

class SymptomChecker:
    """
    Orchestrates the full medical symptom checker flow using ML for smart symptom ordering
    and exact row matching for recommendations.
    """
    def __init__(self, csv_path: str):
        self.predictor = SymptomPredictor()
        self.predictor.train_on_csv(csv_path)
        self.data = pd.read_csv(csv_path) if csv_path.endswith('.csv') else pd.read_excel(csv_path, engine="openpyxl")
        self.demographics = {}
        self.current_symptoms = {}
        self.available_symptoms = SYMPTOMS.copy()
        self.followup_answers = {}
        self.matched_rows = None

    def run_conversation(self):
        """
        Runs the full conversation: demographics, symptom questioning, follow-ups, and final recommendation.
        """
        self.collect_demographics()
        self.smart_symptom_questioning()
        self.ask_followup_questions()
        recommendation = self.find_exact_match()
        print("\nRecommendation:")
        print(recommendation)

    def collect_demographics(self):
        """
        Collects age, weight, and gender, and buckets them as required.
        """
        age = int(input("What is your age (in years)? "))
        self.demographics['Age'] = self.bucket_age(age)
        weight = int(input("What is your weight (in kg)? "))
        self.demographics['Weight'] = self.bucket_weight(weight)
        gender = input("What is your gender? (Male/Female): ").strip().title()
        self.demographics['Gender'] = gender
        print(f"(Internally mapped: Age={self.demographics['Age']}, Weight={self.demographics['Weight']}, Gender={self.demographics['Gender']})")
        # Remove Menstrual Cramps for Male
        if gender.lower() == 'male' and 'Menstrual Cramps' in self.available_symptoms:
            self.available_symptoms.remove('Menstrual Cramps')

    def smart_symptom_questioning(self):
        """
        ML-driven symptom questioning loop. Asks about symptoms in order of predicted probability.
        Updates current_symptoms and available_symptoms.
        """
        print("\nWhich of these symptoms do you have:")
        print(", ".join(self.available_symptoms))
        # Accept symptom input case-insensitively
        available_map = {s.strip().lower(): s for s in self.available_symptoms}
        first = input("Enter one symptom you have (exactly as shown): ").strip().lower()
        while first not in available_map:
            print("Please enter a valid symptom from the list.")
            first = input("Enter one symptom you have (exactly as shown): ").strip().lower()
        first_symptom = available_map[first]
        self.current_symptoms[first_symptom] = "Yes"
        self.available_symptoms.remove(first_symptom)
        
        # Ask about other symptoms using ML 
        while True:
            ranked = self.predictor.get_next_best_symptom(self.current_symptoms, self.available_symptoms, self.demographics)
            if not ranked or ranked[0]['probability'] < 0.1:
                break
            next_symptom = ranked[0]['symptom']
            confidence = ranked[0]['probability']
            ans = input(f"Do you also have {next_symptom}? (Confidence: {confidence:.0%}) (Yes/No): ").strip().lower()
            while ans not in ["yes", "no"]:
                ans = input(f"Please answer Yes or No. Do you also have {next_symptom}? (Yes/No): ").strip().lower()
            self.current_symptoms[next_symptom] = "Yes" if ans == "yes" else "No"
            self.available_symptoms.remove(next_symptom)
        
        # Mark all remaining symptoms as No
        for s in self.available_symptoms:
            self.current_symptoms[s] = "No"
        
        # Now filter the dataset based on ALL collected symptoms at once
        print(f"[DEBUG] Current symptoms: {self.current_symptoms}")
        print(f"[DEBUG] Dataset shape before filtering: {self.data.shape}")
        
        # Let's check what the actual values look like in the dataset
        for symptom in self.current_symptoms.keys():
            unique_values = self.data[symptom].astype(str).str.strip().str.lower().unique()
            print(f"[DEBUG] Unique values for {symptom}: {unique_values}")
        
        self.matched_rows = self.data.copy()
        for s, v in self.current_symptoms.items():
            before_count = len(self.matched_rows)
            # Let's see what we're actually filtering
            condition = self.matched_rows[s].astype(str).str.strip().str.lower() == v.strip().lower()
            matching_rows = self.matched_rows[condition]
            print(f"[DEBUG] Looking for {s}={v}")
            print(f"[DEBUG] Found {len(matching_rows)} rows with {s}={v}")
            if len(matching_rows) > 0:
                print(f"[DEBUG] Sample rows with {s}={v}:")
                print(matching_rows[['Age', 'Gender', 'Weight', s]][:3])
            
            self.matched_rows = matching_rows
            after_count = len(self.matched_rows)
            print(f"[DEBUG] Filtering {s}={v}: {before_count} -> {after_count} rows")
            if self.matched_rows.empty:
                print(f"[DEBUG] No rows left after filtering {s}={v}")
                print("No exact guidance available, please consult a doctor.")
                return
        print(f"[DEBUG] Final filtered dataset shape: {self.matched_rows.shape}")

    def ask_followup_questions(self):
        """
        For each confirmed symptom, asks the follow-up questions from the matched rows.
        Stores all answers for exact matching.
        """
        if self.matched_rows is None or self.matched_rows.empty:
            return
        # Get all unique follow-up questions for current symptoms
        followup_cols = [f"Follow up question {i}" for i in range(1, 5)]
        answer_cols = [f"Answer {i}" for i in range(1, 4)]
        asked = set()
        
        # Collect all valid follow-up questions from matched rows
        valid_questions = []
        for idx, row in self.matched_rows.iterrows():
            for i, fq_col in enumerate(followup_cols):
                fq = row.get(fq_col, None)
                # Skip empty, NaN, or invalid questions (like "–")
                if pd.isna(fq) or not str(fq).strip() or str(fq).strip() == '–':
                    continue
                fq_key = fq.strip().lower()
                if fq_key not in asked:
                    valid_questions.append((i, fq_col, fq, answer_cols[i] if i < 3 else None))
                    asked.add(fq_key)
        
        # Ask valid follow-up questions
        for i, fq_col, fq, ans_col in valid_questions:
            if ans_col:  # Questions 1-3 that expect answers
                ans = input(f"{fq} ").strip()
                self.followup_answers[fq_col] = ans
                self.followup_answers[ans_col] = ans
            else:  # Question 4 (informational only)
                print(fq)
        
        # Filter matched_rows by follow-up answers (case-insensitive)
        for i, fq_col in enumerate(followup_cols[:3]):
            ans_col = answer_cols[i]
            ans = self.followup_answers.get(ans_col, None)
            if ans is not None:
                self.matched_rows = self.matched_rows[self.matched_rows[ans_col].astype(str).str.strip().str.lower() == ans.strip().lower()]

    def find_exact_match(self) -> str:
        """
        Searches for a row that matches all demographics, symptoms, and follow-up answers exactly.
        Returns the OTC/Doc recommendation or fallback message.
        """
        if self.matched_rows is None or self.matched_rows.empty:
            print("[DEBUG] No rows left after initial symptom filtering.")
            return "No exact guidance available, please consult a doctor."
        # Filter by demographics (case-insensitive)
        # Age: match any bucket in the list
        age_buckets = self.demographics['Age']
        before = len(self.matched_rows)
        self.matched_rows = self.matched_rows[self.matched_rows['Age'].astype(str).str.strip().str.lower().isin([b.strip().lower() for b in age_buckets])]
        after = len(self.matched_rows)
        print(f"[DEBUG] Demographic filter: Age in {age_buckets} | Rows before: {before}, after: {after}")
        if after == 0:
            print(f"[DEBUG] No rows left after demographic filter: Age in {age_buckets}")
            return "No exact guidance available, please consult a doctor."
        # Continue with other demographic filters
        for k, v in self.demographics.items():
            if k == 'Age':
                continue
            before = len(self.matched_rows)
            self.matched_rows = self.matched_rows[self.matched_rows[k].astype(str).str.strip().str.lower() == v.strip().lower()]
            after = len(self.matched_rows)
            print(f"[DEBUG] Demographic filter: {k}={v} | Rows before: {before}, after: {after}")
            if after == 0:
                print(f"[DEBUG] No rows left after demographic filter: {k}={v}")
                return "No exact guidance available, please consult a doctor."
        # Filter by all symptoms (case-insensitive)
        for s, v in self.current_symptoms.items():
            before = len(self.matched_rows)
            self.matched_rows = self.matched_rows[self.matched_rows[s].astype(str).str.strip().str.lower() == v.strip().lower()]
            after = len(self.matched_rows)
            print(f"[DEBUG] Symptom filter: {s}={v} | Rows before: {before}, after: {after}")
            if after == 0:
                print(f"[DEBUG] No rows left after symptom filter: {s}={v}")
                return "No exact guidance available, please consult a doctor."
        # Filter by follow-up answers (case-insensitive)
        for i in range(1, 4):
            ans_col = f"Answer {i}"
            ans = self.followup_answers.get(ans_col, None)
            if ans is not None:
                before = len(self.matched_rows)
                self.matched_rows = self.matched_rows[self.matched_rows[ans_col].astype(str).str.strip().str.lower() == ans.strip().lower()]
                after = len(self.matched_rows)
                print(f"[DEBUG] Follow-up filter: {ans_col}={ans} | Rows before: {before}, after: {after}")
                if after == 0:
                    print(f"[DEBUG] No rows left after follow-up filter: {ans_col}={ans}")
                    return "No exact guidance available, please consult a doctor."
        if not self.matched_rows.empty:
            print(f"[DEBUG] Final matched row index: {self.matched_rows.index[0]}")
            return str(self.matched_rows.iloc[0]["OTC/Doc"])
        else:
            print("[DEBUG] No rows left after all filters.")
            return "No exact guidance available, please consult a doctor."

    def bucket_age(self, age: int) -> list:
        # All unique buckets from the dataset
        age_buckets = [
            "18-25", "26-35", "35-50", "50-80",
            "26-36", "26-37", "26-38", "26-39", "26-40",
            "26-41", "26-42", "26-43", "26-44", "26-45", "26-46"
        ]
        matches = []
        for bucket in age_buckets:
            parts = bucket.split('-')
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                if int(parts[0]) <= age <= int(parts[1]):
                    matches.append(bucket)
        return matches

    def bucket_weight(self, weight: int) -> str:
        """
        Maps weight to bucket: 40-60 kg, 60-90 kg, 90+ kg
        """
        if 40 <= weight <= 60:
            return "40-60"
        elif 61 <= weight <= 90:
            return "60-90"
        elif weight > 90:
            return "90+"
        else:
            return "Other" 