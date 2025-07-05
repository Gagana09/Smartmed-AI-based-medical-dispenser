import pandas as pd
from symptom_predictor import SymptomPredictor, SYMPTOMS
from symptom_checker import SymptomChecker

# --- CONFIG ---
DATASET_PATH = r"C:/Users/Supriya S/OneDrive/Desktop/IDP/dataset_1.xlsx"
AGE_BUCKETS = [(18, 25, "18-25"), (26, 35, "26-35"), (36, 50, "36-50"), (51, 80, "50-80")]
WEIGHT_BUCKETS = [(40, 60, "40-60"), (61, 90, "60-90"), (91, 300, ">90")]
GENDER_OPTIONS = ["Male", "Female"]

def bucket(value, buckets):
    for low, high, label in buckets:
        if low <= value <= high:
            return label
    return None

def ask(prompt):
    return input(prompt + "\n> ").strip()

def normalize_title_case(s):
    if isinstance(s, str):
        return s.strip().title()
    return s

def main():
    # Load dataset
    df = pd.read_excel(DATASET_PATH, engine="openpyxl")
    # Normalize relevant columns to title case for case-insensitive matching
    for col in ["Gender", "Age", "Weight"] + SYMPTOMS:
        if col in df.columns:
            df[col] = df[col].apply(normalize_title_case)

    predictor = SymptomPredictor()
    predictor.train_on_csv(DATASET_PATH)

    print("Hello! I'm your SmartMed Assistant 🤖. Let's find out the right treatment for you.")

    # Demographics
    while True:
        try:
            age = int(ask("What is your age? (18–80)"))
            age_bucket = bucket(age, AGE_BUCKETS)
            if not age_bucket:
                print("Sorry, age not supported. Please consult a doctor.")
                return
            break
        except ValueError:
            print("Please enter a valid number.")
    while True:
        try:
            weight = float(ask("What is your weight (in kg)? (40–300)"))
            weight_bucket = bucket(weight, WEIGHT_BUCKETS)
            if not weight_bucket:
                print("Weight out of supported range. Please consult a doctor.")
                return
            break
        except ValueError:
            print("Please enter a valid number.")
    while True:
        gender = ask("What is your gender? (Male/Female)").title()
        if gender in GENDER_OPTIONS:
            break
        print("Please enter 'Male' or 'Female'.")

    # Main symptom
    main_symptom = None
    while not main_symptom:
        user_symptom = ask("Which of these symptoms do you have: " + ", ".join(SYMPTOMS if gender != "Male" else [s for s in SYMPTOMS if s != "Menstrual Cramps"]))
        for s in SYMPTOMS:
            if s.lower() in user_symptom.lower():
                main_symptom = s
                break
        if not main_symptom:
            print("Sorry, please enter a main symptom from the list.")

    # Filter dataset by demographics (case-insensitive)
    symptom_list = SYMPTOMS if gender != "Male" else [s for s in SYMPTOMS if s != "Menstrual Cramps"]
    filter_df = df[
        (df["Gender"] == gender) &
        (df["Age"] == age_bucket) &
        (df["Weight"] == weight_bucket)
    ]
    if filter_df.empty:
        print("Sorry, no matching profile found. Please consult a doctor.")
        return
    user_flags = {s: "No" for s in symptom_list}
    user_flags[main_symptom] = "Yes"
    asked = set([main_symptom])
    filter_df = filter_df[filter_df[main_symptom] == "Yes"]
    if filter_df.empty:
        print("Sorry, no matching profile found. Please consult a doctor.")
        return

    # ML-powered symptom questioning
    while True:
        if len(filter_df) == 1:
            break
        unasked = [s for s in symptom_list if s not in asked]
        if not unasked:
            break
        ranked = predictor.get_next_symptom_priority(user_flags, unasked)
        ranked = [(s, p) for s, p in ranked if p >= 0.1]
        if not ranked:
            break
        next_sym, prob = ranked[0]
        ans = ask(f"Do you also have {next_sym}? (Yes/No) [Confidence: {prob*100:.1f}%]").title()
        while ans not in ["Yes", "No"]:
            ans = ask(f"Please answer Yes or No. Do you have {next_sym}?").title()
        user_flags[next_sym] = ans
        asked.add(next_sym)
        filter_df = filter_df[filter_df[next_sym] == ans]
        if filter_df.empty:
            print("Sorry, no matching profile found. Please consult a doctor.")
            return

    # Row matching for final recommendation
    match = filter_df.copy()
    for s in symptom_list:
        match = match[match[s] == user_flags[s]]
    if match.empty:
        print("Sorry, no matching profile found. Please consult a doctor.")
        return
    row = match.iloc[0]

    # Follow-up questions - only ask valid questions (skip empty or "–")
    valid_questions_asked = 0
    for i in range(1, 5):
        q_col = f"Follow up question {i}"
        a_col = f"Answer {i}"
        q = row[q_col] if q_col in row and pd.notna(row[q_col]) and str(row[q_col]).strip() != "" and str(row[q_col]).strip() != "–" else None
        a = row[a_col] if a_col in row and pd.notna(row[a_col]) and str(row[a_col]).strip() != "" else None
        
        if q:  # Only process if question is valid
            valid_questions_asked += 1
            if i == 1:
                print(f"{q}\nAnswer: {a if a else ''}")
            elif i == 2:
                print(f"{q}\nAnswer: {a if a else ''}")
            elif i in [3, 4]:
                print(q)
                if i == 3:
                    user_a = ask("Your answer:")
        elif valid_questions_asked == 0 and i == 2:
            # If no valid questions found by question 2, break and go to recommendation
            break

    # Final recommendation
    print("\nBased on your profile and symptoms:")
    print("→ Recommendation:", row["OTC/Doc"] if "OTC/Doc" in row else row["Otc/Doc"])

class MedicalChatbot:
    def __init__(self, csv_path='dataset_1.xlsx'):
        self.checker = SymptomChecker(csv_path)
        self.state = {
            'step': 'demographics',
            'demographics': {},
            'symptoms': {},
            'available_symptoms': self.checker.available_symptoms.copy(),
            'followup_answers': {},
            'asked_followups': set(),
            'matched_rows': None,
            'last_question': None,
            'symptom_queue': [],
        }
        self._demographic_questions = [
            'What is your age (in years)?',
            'What is your weight (in kg)?',
            'What is your gender? (Male/Female):'
        ]
        self._demographic_keys = ['Age', 'Weight', 'Gender']
        self._demographic_index = 0
        self._symptom_first = True
        self._symptom_ranking = []
        self._followup_index = 0
        self._final_recommendation = None

    def get_response(self, user_input):
        # Demographics collection
        if self.state['step'] == 'demographics':
            greetings = {'hi', 'hello', 'hey', 'greetings', ''}
            if self._demographic_index < 3:
                key = self._demographic_keys[self._demographic_index]
                if user_input.strip().lower() in greetings:
                    return self._demographic_questions[self._demographic_index]
                if key == 'Age':
                    try:
                        age = int(user_input)
                        age_bucket = None
                        for low, high, label in AGE_BUCKETS:
                            if low <= age <= high:
                                age_bucket = label
                                break
                        if not age_bucket:
                            return "Sorry, age not supported. Please consult a doctor."
                        self.state['demographics']['Age'] = [age_bucket]
                        self._demographic_index += 1
                        return self._demographic_questions[self._demographic_index]
                    except:
                        return 'Please enter a valid age (number).'
                elif key == 'Weight':
                    try:
                        weight = float(user_input)
                        WEIGHT_BUCKETS = [(40, 60, "40-60"), (61, 90, "60-90"), (91, 300, ">90")]
                        weight_bucket = None
                        for low, high, label in WEIGHT_BUCKETS:
                            if low <= weight <= high:
                                weight_bucket = label
                                break
                        if not weight_bucket:
                            return "Weight out of supported range. Please consult a doctor."
                        self.state['demographics']['Weight'] = weight_bucket
                        self._demographic_index += 1
                        return self._demographic_questions[self._demographic_index]
                    except:
                        return 'Please enter a valid weight (number).'
                elif key == 'Gender':
                    gender = user_input.strip().title()
                    if gender not in ["Male", "Female"]:
                        return "Please enter 'Male' or 'Female'."
                    self.state['demographics']['Gender'] = gender
                    if gender.lower() == 'male' and 'Menstrual Cramps' in self.state['available_symptoms']:
                        self.state['available_symptoms'].remove('Menstrual Cramps')
                    self._demographic_index += 1
                    self.state['step'] = 'symptom_questioning'
                    return f"Which of these symptoms do you have: {', '.join(self.state['available_symptoms'])}"
            else:
                self.state['step'] = 'symptom_questioning'
                return f"Which of these symptoms do you have: {', '.join(self.state['available_symptoms'])}"

        # Symptom questioning
        if self.state['step'] == 'symptom_questioning':
            if self._symptom_first:
                # First symptom
                available_map = {s.strip().lower(): s for s in self.state['available_symptoms']}
                first = user_input.strip().lower()
                if first not in available_map:
                    return f"Please enter a valid symptom from the list: {', '.join(self.state['available_symptoms'])}"
                first_symptom = available_map[first]
                self.state['symptoms'][first_symptom] = 'Yes'
                self.state['available_symptoms'].remove(first_symptom)
                # Filter dataset to rows where first symptom is Yes
                self.checker.matched_rows = self.checker.data[self.checker.data[first_symptom].astype(str).str.strip().str.lower() == 'yes']
                self._symptom_first = False
            else:
                # Answer to ML-prompted symptom
                last_symptom = self._symptom_ranking[0]['symptom'] if self._symptom_ranking else None
                ans = user_input.strip().lower()
                if ans not in ['yes', 'no']:
                    return f"Please answer Yes or No. Do you also have {last_symptom}?"
                self.state['symptoms'][last_symptom] = 'Yes' if ans == 'yes' else 'No'
                self.state['available_symptoms'].remove(last_symptom)
                # Update matched_rows
                for s, v in self.state['symptoms'].items():
                    self.checker.matched_rows = self.checker.matched_rows[self.checker.matched_rows[s].astype(str).str.strip().str.lower() == v.strip().lower()]
                if self.checker.matched_rows.empty:
                    self.state['step'] = 'done'
                    return "No exact guidance available, please consult a doctor."
            # ML symptom ranking
            ranked = self.checker.predictor.get_next_best_symptom(self.state['symptoms'], self.state['available_symptoms'], self.state['demographics'])
            self._symptom_ranking = ranked
            if not ranked or ranked[0]['probability'] < 0.1:
                # Mark all remaining symptoms as No
                for s in self.state['available_symptoms']:
                    self.state['symptoms'][s] = 'No'
                for s, v in self.state['symptoms'].items():
                    self.checker.matched_rows = self.checker.matched_rows[self.checker.matched_rows[s].astype(str).str.strip().str.lower() == v.strip().lower()]
                self.state['step'] = 'followup'
                self._followup_index = 0
                return self._next_followup_question()
            next_symptom = ranked[0]['symptom']
            confidence = ranked[0]['probability']
            return f"Do you also have {next_symptom}? (Confidence: {confidence:.0%}) (Yes/No):"

        # Follow-up questions
        if self.state['step'] == 'followup':
            # Store answer to previous follow-up
            if self.state['last_question']:
                self.state['followup_answers'][self.state['last_question']] = user_input.strip()
            # Ask next follow-up
            q = self._next_followup_question()
            if q:
                return q
            else:
                self.state['step'] = 'recommendation'
                return self.get_recommendation()

        # Recommendation
        if self.state['step'] == 'recommendation':
            return self.get_recommendation()

        # Done
        if self.state['step'] == 'done':
            return "No exact guidance available, please consult a doctor."

        return "Sorry, I didn't understand."

    def _next_followup_question(self):
        # Get all unique follow-up questions from matched rows
        if self.checker.matched_rows is None or self.checker.matched_rows.empty:
            self.state['step'] = 'done'
            return None
        followup_cols = [f"Follow up question {i}" for i in range(1, 5)]
        answer_cols = [f"Answer {i}" for i in range(1, 4)]
        asked = self.state['asked_followups']
        
        # Collect all valid follow-up questions from matched rows
        valid_questions = []
        for idx, row in self.checker.matched_rows.iterrows():
            for i, fq_col in enumerate(followup_cols):
                fq = row.get(fq_col, None)
                # Skip empty, NaN, or invalid questions (like "–")
                if pd.isna(fq) or not str(fq).strip() or str(fq).strip() == '–':
                    continue
                fq_key = fq.strip().lower()
                if fq_key not in asked:
                    valid_questions.append((i, fq_col, fq, answer_cols[i] if i < 3 else None))
                    asked.add(fq_key)
        
        # Return the first valid question
        if valid_questions:
            i, fq_col, fq, ans_col = valid_questions[0]
            self.state['last_question'] = ans_col
            return fq
        
        # After all follow-ups
        self.state['last_question'] = None
        return None

    def get_recommendation(self):
        # Store follow-up answers in checker
        self.checker.followup_answers = self.state['followup_answers']
        rec = self.checker.find_exact_match()
        self.state['step'] = 'done'
        model_type = self.checker.predictor.model_type
        return f"{rec}\n[ML Model Used: {model_type}]"

    def get_greeting(self):
        return "Hello! I'm your medical assistant. Please tell me your age (in years) to get started."

    def get_serializable_state(self):
        # Only needed for final storage, not for every step
        serializable_state = self.state.copy()
        if 'asked_followups' in serializable_state and isinstance(serializable_state['asked_followups'], set):
            serializable_state['asked_followups'] = list(serializable_state['asked_followups'])
        return serializable_state

    def restore_state(self, state):
        # Only needed for initial session setup, not for every step
        self.state = state
        if 'asked_followups' in self.state and isinstance(self.state['asked_followups'], list):
            self.state['asked_followups'] = set(self.state['asked_followups'])

if __name__ == "__main__":
    bot_instance = MedicalChatbot(csv_path='dataset_1.xlsx')
    main() 