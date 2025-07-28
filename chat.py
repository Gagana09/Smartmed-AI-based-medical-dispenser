import pandas as pd
from symptom_predictor import SymptomPredictor, SYMPTOMS
from symptom_checker import SymptomChecker
import json
from datetime import datetime

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

    predictor = SymptomPredictor(real_time_learning=True)  # Enable real-time learning
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
        ranked = predictor.get_next_best_symptom(user_flags, unasked, {"Age": age_bucket, "Gender": gender, "Weight": weight_bucket})
        ranked = [(s['symptom'], s['probability']) for s in ranked if s['probability'] >= 0.1]
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
        question_col = f"Follow up question {i}"
        if question_col in row and pd.notna(row[question_col]) and str(row[question_col]).strip() not in ["", "–", "nan"]:
            question = str(row[question_col]).strip()
            answer = ask(f"{question} (Yes/No)")
            while answer.title() not in ["Yes", "No"]:
                answer = ask("Please answer Yes or No.")
            valid_questions_asked += 1
        if valid_questions_asked >= 2:  # Limit to 2 follow-up questions
            break

    # Final recommendation
    recommendation = row.get("Recommendation", "Please consult a doctor for proper diagnosis.")
    print(f"\nBased on your symptoms, I recommend:\n{recommendation}")
    
    # Collect feedback for real-time learning
    try:
        rating = int(ask("\nHow helpful was this recommendation? (1-5, where 5 is very helpful): "))
        if 1 <= rating <= 5:
            # Add interaction to real-time learning system
            demographics = {"Age": age_bucket, "Gender": gender, "Weight": weight_bucket}
            predictor.add_user_interaction(
                demographics=demographics,
                symptoms=user_flags,
                final_recommendation=recommendation,
                success_rating=rating
            )
            print("Thank you for your feedback! This helps improve the system.")
        else:
            print("Invalid rating. Feedback not recorded.")
    except ValueError:
        print("Invalid input. Feedback not recorded.")
    
    # Show learning stats
    stats = predictor.get_learning_stats()
    print(f"\nLearning System Stats:")
    print(f"- Total interactions: {stats['total_interactions']}")
    print(f"- Model type: {stats['model_type']}")
    print(f"- Real-time learning: {'Enabled' if stats['real_time_learning_enabled'] else 'Disabled'}")

class MedicalChatbot:
    def __init__(self, csv_path='dataset_1.xlsx'):
        self.df = pd.read_excel(csv_path, engine="openpyxl")
        # Normalize relevant columns to title case for case-insensitive matching
        for col in ["Gender", "Age", "Weight"] + SYMPTOMS:
            if col in self.df.columns:
                self.df[col] = self.df[col].apply(normalize_title_case)
        
        # Initialize with real-time learning enabled
        self.predictor = SymptomPredictor(real_time_learning=True)
        self.predictor.train_on_csv(csv_path)
        
        self.checker = SymptomChecker(csv_path)
        self.checker.predictor = self.predictor  # Use the same predictor instance
        
        # State management
        self.state = {
            'step': 'demographics',
            'demographics': {},
            'symptoms': {},
            'available_symptoms': SYMPTOMS.copy(),
            'asked_followups': set(),
            'followup_answers': {},
            'last_question': None,
            'final_recommendation': None,
            'session_start_time': datetime.now().isoformat()
        }
        
        # Real-time learning tracking
        self.interaction_data = {
            'session_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'questions_asked': [],
            'symptom_confidence_scores': [],
            'final_outcome': None
        }
        
        self._demographic_questions = [
            "What is your age? (18-80)",
            "What is your weight (in kg)? (40-300)",
            "What is your gender? (Male/Female)"
        ]
        self._demographic_keys = ['Age', 'Weight', 'Gender']
        self._demographic_index = 0
        self._symptom_first = True
        self._symptom_ranking = []
        self._followup_index = 0
        self._final_recommendation = None

    def get_response(self, user_input):
        # Track interaction for learning
        self._track_interaction(user_input)
        
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
                        self.state['demographics']['Age'] = age_bucket
                        self._demographic_index += 1
                        return self._demographic_questions[self._demographic_index]
                    except:
                        return 'Please enter a valid age (number).'
                elif key == 'Weight':
                    try:
                        weight = float(user_input)
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
            
            # Track confidence scores for learning
            if ranked:
                self.interaction_data['symptom_confidence_scores'].append({
                    'symptom': ranked[0]['symptom'],
                    'confidence': ranked[0]['probability'],
                    'timestamp': datetime.now().isoformat()
                })
            
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

    def _track_interaction(self, user_input):
        """Track user interactions for real-time learning"""
        self.interaction_data['questions_asked'].append({
            'input': user_input,
            'step': self.state['step'],
            'timestamp': datetime.now().isoformat()
        })

    def _next_followup_question(self):
        # Get all unique follow-up questions from matched rows
        if self.checker.matched_rows is None or self.checker.matched_rows.empty:
            self.state['step'] = 'done'
            return None
        followup_cols = [f"Follow up question {i}" for i in range(1, 5)]
        answer_cols = [f"Answer {i}" for i in range(1, 4)]
        asked = self.state['asked_followups']
        
        # Find next unasked follow-up question
        for col in followup_cols:
            if col in self.checker.matched_rows.columns:
                unique_questions = self.checker.matched_rows[col].dropna().unique()
                for question in unique_questions:
                    if question not in asked and str(question).strip() not in ["", "–", "nan"]:
                        self.state['asked_followups'].add(question)
                        self.state['last_question'] = question
                        return str(question).strip()
        
        # No more follow-up questions
        self.state['step'] = 'recommendation'
        return None

    def get_recommendation(self):
        # Store follow-up answers in checker
        self.checker.followup_answers = self.state['followup_answers']
        
        # Get recommendation
        recommendation = self.checker.get_recommendation()
        self.state['final_recommendation'] = recommendation
        
        # Add interaction to real-time learning system
        self.predictor.add_user_interaction(
            demographics=self.state['demographics'],
            symptoms=self.state['symptoms'],
            final_recommendation=recommendation
        )
        
        # Update interaction data
        self.interaction_data['final_outcome'] = {
            'recommendation': recommendation,
            'symptoms': self.state['symptoms'],
            'demographics': self.state['demographics'],
            'timestamp': datetime.now().isoformat()
        }
        
        return recommendation

    def get_greeting(self):
        return "Hello! I'm your SmartMed Assistant 🤖. Let's find out the right treatment for you. What is your age? (18-80)"

    def get_serializable_state(self):
        # Only needed for final storage, not for every step
        return {
            'session_id': self.interaction_data['session_id'],
            'final_recommendation': self.state['final_recommendation'],
            'learning_stats': self.predictor.get_learning_stats()
        }

    def restore_state(self, state):
        # Only needed for initial session setup, not for every step
        pass

if __name__ == "__main__":
    bot_instance = MedicalChatbot(csv_path='dataset_1.xlsx')
    main() 