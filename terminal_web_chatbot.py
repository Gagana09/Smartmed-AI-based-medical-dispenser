import pandas as pd
from symptom_predictor import SymptomPredictor, SYMPTOMS
import threading
import re
import difflib
import string

DATASET_PATH = "dataset_1.xlsx"
AGE_BUCKETS = [(18, 25, "18-25"), (26, 35, "26-35"), (36, 50, "36-50"), (51, 80, "50-80")]
WEIGHT_BUCKETS = [(40, 60, "40-60"), (61, 90, "60-90"), (91, 300, ">90")]
GENDER_OPTIONS = ["Male", "Female"]

# Global cache for dataset and predictor
_GLOBAL_DF = None
_GLOBAL_PREDICTOR = None
_GLOBAL_LOCK = threading.Lock()

# List of medicines to check for dispensing prompt
MEDICINE_LIST = [
    'Paracetamol 500 mg',
    'Vitamin C (Limcee)',
    'Paracetamol',
    'ORS',
    'Zincovit syrup/tab',
    'Vitamin C',
    'Limcee 500 mg (Vit C)',
    'Zincovit',
    'electrolyte solution',
    'electrolyte water',
    'antihistamines (e.g. Cetirizine)',
    'Lozenges (e.g. Strepsils)',
    'Decongestant (e.g. Phenylephrine)',
    'Cetirizine',
    'Saline nasal spray',
    'Lozenges',
    'Antihistamines',
    'mild saline spray',
    'nasal drops',
    'Benadryl Dry Cough',
    'Honey-ginger lozenges',
    'Mucosolvan / Ambrolite',
    'nasal decongestant (e.g. oxymetazoline)',
    'saline nasal drops',
    'Meftal Spas',
    'mild nasal spray',
    'Pantoprazole',
    'Antacids like Gelusil',
    'Simethicone (for gas)',
    'Clove oil (topical)',
    'Paracetamol 500mg',
    'Paracetamol',
    'lozenges',
    'Caffeine + Paracetamol',
    'Levocetirizine 5 mg',
    'Bromhexine syrup',
    'Paracetamol',
    'Ambroxol/Bromhexine syrup',
    'saline nasal spray',
    'lozenges',
    'herbal lozenges (e.g., Vicks Lozenges/Adulsa)',
    'Paracetamol combo tablet',
    'Paracetamol 500–650 mg',
    'Caffeine + Paracetamol combo (e.g., Saridon)',
    'OTC analgesics',
    'Herbal lozenge (Vicks/Himalaya)',
    'Herbal lozenges',
    'dextromethorphan',
    'Caffeine + Paracetamol combo tablet',
    'cough lozenges (e.g., benzydamine or ambroxol lozenges',
    'Herbal lozenge (Vicks/Himalaya)',
    'dextromethorphan syrup',
    'Herbal lozenges',
    'paracetamol 500 mg',
    'Meftal-Spas 250–500 mg',
    'antacid (e.g., Rantac)',
    'Paracetamol 650 mg',
    'clove/benzocaine gel + warm rinse',
    'warm saline rinse + topical clove/benzocaine gel',
    'clove oil or dental pain gel',
    'antacid',
    'mouthwash (chlorhexidine if available)',
    'H2-blocker or antacid (Rantac, Gelusil)',
    'Pantoprazole 40 mg',
    'Antacid (Digene or Gelusil)',
    'warm rinse + clove gel',
    'Gelusil or Digene',
    'clove or antiseptic gel',
    'pain relief gel (e.g., diclofenac)',
    'crepe bandage',
    'cold pack',
    'compression wrap',
    'compression bandage',
    'elastic support bandage',
]

# Updated mapping for follow-up 1 questions and their options (full list from user)
FOLLOWUP_1_QUESTIONS = {
    "Is your fever above 104°F?": ["Yes", "No"],
    "Do you have a runny or blocked nose?": ["Runny", "Blocked"],
    "Is the cough dry or brings mucus?": ["Dry", "Mucus"],
    "Are the cramps interfering with daily activities?": ["Yes", "No"],
    "Have you taken any spicy, oily or heavy food?": ["Yes", "No"],
    "Is the pain sharp": ["Yes", "No"],
    "Is the pain constant?": ["Yes", "No"],
    "Is the pain in one specific area?": ["Yes", "No"],
    "Is there any swelling or bruising at the injured area?": ["Yes", "No"],
    "Is the headache throbbing or dull?": ["Throbbing", "Dull"],
    "Is pain spread throughout your body?": ["Yes", "No"],
    "Are you experiencing nasal congestion?": ["Yes", "No"],
    "Is the headache worse around your forehead or eyes?": ["Forehead", "Eyes"],
    "Does coughing make the headache worse?": ["Yes", "No"],
    "Do you feel muscle pain from coughing too much?": ["Yes", "No"],
    "Is the pain also in your lower back or thighs?": ["Lower Back", "Thighs"],
    "Do you feel tired or weak along with both symptoms?": ["Yes", "No"],
    "Do you feel nausea or bloating with the headache?": ["Yes", "No"],
    "Is there swelling near the tooth or gum?": ["Yes", "No"],
    "Is the cough dry or with mucus?": ["Dry", "Mucus"],
}

# Mapping for follow-up 2 questions and their options
FOLLOWUP_2_QUESTIONS = {
    "Are you feeling weak or tired along with the fever": ["Yes", "No"],
    "Do you have a sore throat": ["Yes", "No"],
    "Have you been exposed to dust/smoke/allergens": ["Yes", "No"],
    "Is this pain typical for your periods": ["Yes", "No"],
    "Is the pain burning, bloating or cramping": ["Yes", "No"],
    "Does it get worse with hot, cold, sweet foods": ["Yes", "No"],
    "Did you get enough sleep, had food or stay hydrated": ["Yes", "No"],
    "Have you done any physical activity recently": ["Yes", "No"],
    "Can you move or put weight on the affected part": ["Yes", "No"],
    "Are you feeling chills or sweating": ["Chills", "Sweating"],
    "Do you feel tired or weak along with cough": ["Yes", "No"],
    "Do you feel light sensitivity or eye strain": ["Yes", "No"],
    "Do you feel shivering or fatigue": ["Yes", "No"],
    "Does the cough worsen at night": ["Yes", "No"],
    "Does bending forward make headache worse": ["Yes", "No"],
    "Is your throat sore or irritated": ["Yes", "No"],
    "Is breathing deeply painful": ["Yes", "No"],
    "Do you feel bloated or fatigued": ["Bloated", "Fatigued"],
    "Did you strain yourself or have poor sleep recently": ["Yes", "No"],
    "Do the symptoms feel worse after eating or skipping meals": ["Yes", "No"],
    "Do you have a bad taste or foul smell in your mouth": ["Yes", "No"],
}

# Mapping for follow-up 3 questions and their options (add your questions and options here)
FOLLOWUP_3_QUESTIONS = {
    # Example:
    # "Is your appetite reduced": ["Yes", "No"],
    # Add your actual follow-up 3 questions and options here
}

def filter_longest_medicines(medicine_list):
    # Normalize to lower and strip for comparison, but keep original for output
    normalized = [(med.strip().lower(), med) for med in medicine_list]
    # Sort by length of normalized name descending
    sorted_meds = sorted(normalized, key=lambda x: len(x[0]), reverse=True)
    filtered = []
    seen = set()
    for i, (norm_med, orig_med) in enumerate(sorted_meds):
        if any(norm_med in other for other, _ in sorted_meds[:i]):
            continue
        # Avoid duplicates (case-insensitive)
        if norm_med not in seen:
            filtered.append(orig_med)
            seen.add(norm_med)
    return filtered

def get_global_df_and_predictor():
    global _GLOBAL_DF, _GLOBAL_PREDICTOR
    with _GLOBAL_LOCK:
        if _GLOBAL_DF is None:
            _GLOBAL_DF = pd.read_excel(DATASET_PATH, engine="openpyxl")
            for col in ["Gender", "Age", "Weight"] + SYMPTOMS:
                if col in _GLOBAL_DF.columns:
                    _GLOBAL_DF[col] = _GLOBAL_DF[col].apply(lambda s: s.strip().title() if isinstance(s, str) else s)
        if _GLOBAL_PREDICTOR is None:
            _GLOBAL_PREDICTOR = SymptomPredictor(real_time_learning=True)  # Enable real-time learning
            _GLOBAL_PREDICTOR.train_on_csv(DATASET_PATH)
    return _GLOBAL_DF, _GLOBAL_PREDICTOR

def normalize_question(q):
    # Lowercase, strip, remove punctuation and extra spaces
    q = q.strip().lower()
    q = re.sub(r'[\s]+', ' ', q)
    q = re.sub(r'[?.,:;!]', '', q)
    return q

class TerminalStyleWebChatbot:
    def __init__(self, csv_path=DATASET_PATH):
        self.df, self.predictor = get_global_df_and_predictor()
        self.state = {
            'step': 'ask_age',
            'age': None,
            'age_bucket': None,
            'weight': None,
            'weight_bucket': None,
            'gender': None,
            'symptom_list': None,
            'main_symptom': None,
            'user_flags': {},
            'asked': set(),
            'filter_df': None,
            'ml_symptom': None,
            'followup_index': 1,
            'final_row': None,
        }

    def get_response(self, user_input):
        s = self.state
        if s['step'] == 'ask_age':
            try:
                age = int(user_input)
                age_buckets = self.bucket_age(age)
                if not age_buckets:
                    return "Sorry, age not supported. Please consult a doctor."
                s['age'] = age
                s['age_bucket'] = age_buckets
                s['step'] = 'ask_weight'
                return "What is your weight (in kg)?"
            except:
                return "Please enter a valid age (number)."
        if s['step'] == 'ask_weight':
            try:
                weight = float(user_input)
                weight_bucket = None
                for low, high, label in WEIGHT_BUCKETS:
                    if low <= weight <= high:
                        weight_bucket = label
                        break
                if not weight_bucket:
                    return "Weight out of supported range. Please consult a doctor."
                s['weight'] = weight
                s['weight_bucket'] = weight_bucket
                s['step'] = 'ask_gender'
                return "What is your gender? (Male/Female)"
            except:
                return "Please enter a valid weight (number)."
        if s['step'] == 'ask_gender':
            gender = user_input.strip().title()
            if gender not in GENDER_OPTIONS:
                return "Please enter 'Male' or 'Female'."
            s['gender'] = gender
            s['symptom_list'] = SYMPTOMS if gender != "Male" else [sym for sym in SYMPTOMS if sym != "Menstrual Cramps"]
            s['step'] = 'ask_main_symptom'
            return "Which of these symptoms do you have: " + ", ".join(s['symptom_list'])
        if s['step'] == 'ask_main_symptom':
            user_symptom = user_input.strip()
            main_symptom = None
            for sym in s['symptom_list']:
                if sym.lower() in user_symptom.lower():
                    main_symptom = sym
                    break
            if not main_symptom:
                return "Sorry, please enter a main symptom from the list: " + ", ".join(s['symptom_list'])
            s['main_symptom'] = main_symptom
            s['user_flags'] = {sym: "No" for sym in s['symptom_list']}
            s['user_flags'][main_symptom] = "Yes"
            s['asked'] = set([main_symptom])
            s['step'] = 'ml_symptom_loop'
            # Immediately ask the first related symptom with options
            # Find next unasked symptom
            unasked = [sym for sym in s['symptom_list'] if sym not in s['asked']]
            demographics = {}
            if s.get('age_bucket') is not None:
                demographics['Age'] = s['age_bucket']
            if s.get('weight_bucket') is not None:
                demographics['Weight'] = s['weight_bucket']
            if s.get('gender') is not None:
                demographics['Gender'] = s['gender']
            ranked = self.predictor.get_next_best_symptom(s['user_flags'], unasked, demographics)
            ranked = [r for r in ranked if r['probability'] >= 0.1]
            if not ranked:
                s['step'] = 'filter_and_followup'
                return self._filter_and_followup()
            s['ml_symptom'] = ranked[0]['symptom']
            return {"response": f"Do you also have {s['ml_symptom']}? (Yes/No)", "options": ["Yes", "No"]}
        if s['step'] == 'ml_symptom_loop':
            # Handle ML symptom answer
            if s['ml_symptom']:
                ans = user_input.strip().title()
                if ans not in ["Yes", "No"]:
                    return {
                        "response": f"Please answer Yes or No. Do you also have {s['ml_symptom']}?",
                        "options": ["Yes", "No"]
                    }
                s['user_flags'][s['ml_symptom']] = ans
                s['asked'].add(s['ml_symptom'])
                
                # 🔄 REMOVED: No longer learning after each question - only after complete conversation
                s['ml_symptom'] = None  # Clear the current symptom
                
                # 🔄 NEW: Check if user has confirmed 2 symptoms with "Yes"
                yes_count = sum(1 for v in s['user_flags'].values() if v == "Yes")
                if yes_count >= 2:
                    print(f"🧠 ML Decision: Stopping questions - user has confirmed {yes_count} symptoms")
                    s['step'] = 'filter_and_followup'
                    return self._filter_and_followup()
                
            # Find next unasked symptom
            unasked = [sym for sym in s['symptom_list'] if sym not in s['asked']]
            if not unasked:
                # All symptoms collected, now filter the dataset
                s['step'] = 'filter_and_followup'
                return self._filter_and_followup()
            # Prepare demographics dict if available
            demographics = {}
            if s.get('age_bucket') is not None:
                demographics['Age'] = s['age_bucket']
            if s.get('weight_bucket') is not None:
                demographics['Weight'] = s['weight_bucket']
            if s.get('gender') is not None:
                demographics['Gender'] = s['gender']
            ranked = self.predictor.get_next_best_symptom(s['user_flags'], unasked, demographics)
            ranked = [r for r in ranked if r['probability'] >= 0.1]
            if not ranked:
                # No more symptoms to ask, filter the dataset
                s['step'] = 'filter_and_followup'
                return self._filter_and_followup()
            s['ml_symptom'] = ranked[0]['symptom']
            return {"response": f"Do you also have {s['ml_symptom']}? (Yes/No)", "options": ["Yes", "No"]}
        if s['step'] == 'filter_and_followup':
            return self._filter_and_followup()
        if s['step'] == 'followup_loop':
            if s['filter_df'] is None:
                s['step'] = 'filter_and_followup'
                # Save the pending follow-up answer to process after recompute
                if s.get('current_followup_col'):
                    s['_pending_followup_answer'] = (s['current_followup_col'], user_input.strip())
                return self._filter_and_followup()
            if s['filter_df'] is not None:
                if s.get('current_followup_col'):
                    ans = user_input.strip()
                    col = s['current_followup_col']
                    if col not in s['filter_df'].columns:
                        s['current_followup_col'] = None
                        s['current_followup_qcol'] = None
                        s['current_row_index'] = None
                        for idx, row in s['filter_df'].iterrows():
                            for i in range(1, 4):
                                q_col = f"Follow up question {i}"
                                a_col = f"Answer {i}"
                                if q_col in row and pd.notna(row[q_col]) and str(row[q_col]).strip() != "" and f"followup_{i}_answered" not in s:
                                    s[f"current_followup_col"] = a_col
                                    s[f"current_followup_qcol"] = q_col
                                    s[f"current_row_index"] = idx
                                    q = str(row[q_col]).strip()
                                    norm_q = normalize_question(q)
                                    for d in (FOLLOWUP_1_QUESTIONS, FOLLOWUP_2_QUESTIONS, FOLLOWUP_3_QUESTIONS):
                                        for key in d:
                                            if normalize_question(key) == norm_q:
                                                return {"response": q, "options": d[key]}
                                    return {"response": q, "options": ["Yes", "No"]}
                        if s['filter_df'] is not None and len(s['filter_df']) >= 1:
                            s['final_row'] = s['filter_df'].iloc[0]
                            s['step'] = 'final_recommendation'
                            return self._final_recommendation()
                        else:
                            s['step'] = 'done'
                            return "Sorry, no matching profile found. Please consult a doctor."
                    ans = user_input.strip()
                    col = s['current_followup_col']
                    if col not in s['filter_df'].columns:
                        s['current_followup_col'] = None
                        s['current_followup_qcol'] = None
                        s['current_row_index'] = None
                        for idx, row in s['filter_df'].iterrows():
                            for i in range(1, 4):
                                q_col = f"Follow up question {i}"
                                a_col = f"Answer {i}"
                                if q_col in row and pd.notna(row[q_col]) and str(row[q_col]).strip() != "" and f"followup_{i}_answered" not in s:
                                    s[f"current_followup_col"] = a_col
                                    s[f"current_followup_qcol"] = q_col
                                    s[f"current_row_index"] = idx
                                    q = str(row[q_col]).strip()
                                    norm_q = normalize_question(q)
                                    for d in (FOLLOWUP_1_QUESTIONS, FOLLOWUP_2_QUESTIONS, FOLLOWUP_3_QUESTIONS):
                                        for key in d:
                                            if normalize_question(key) == norm_q:
                                                return {"response": q, "options": d[key]}
                                    return {"response": q, "options": ["Yes", "No"]}
                        if s['filter_df'] is not None and len(s['filter_df']) >= 1:
                            s['final_row'] = s['filter_df'].iloc[0]
                            s['step'] = 'final_recommendation'
                            return self._final_recommendation()
                        else:
                            s['step'] = 'done'
                            return "Sorry, no matching profile found. Please consult a doctor."
                    print(f"[DEBUG] Processing follow-up answer: {col} = {ans}")
                    unique_values = s['filter_df'][col].astype(str).str.strip().str.lower().unique()
                    ans_lower = ans.strip().lower()
                    matching_rows = s['filter_df'][s['filter_df'][col].astype(str).str.strip().str.lower() == ans_lower]
                    if len(matching_rows) == 0:
                        print(f"[DEBUG] No exact match, trying partial matching")
                        for val in unique_values:
                            if ans_lower in val or val in ans_lower:
                                print(f"[DEBUG] Partial match found: '{ans_lower}' matches '{val}'")
                                ans_lower = val
                                matching_rows = s['filter_df'][s['filter_df'][col].astype(str).str.strip().str.lower() == ans_lower]
                                break
                    try:
                        followup_num = int(col.split()[-1])
                    except Exception:
                        followup_num = 1  # fallback
                    s[f"followup_{followup_num}_answer"] = ans_lower
                    s[f"followup_{followup_num}_answered"] = True
                    s['current_followup_col'] = None
                    s['current_followup_qcol'] = None
                    s['current_row_index'] = None
                    s['filter_df'] = matching_rows
                    if s['filter_df'].empty:
                        s['step'] = 'done'
                        return "Sorry, no matching profile found. Please consult a doctor."
                    if len(s['filter_df']) == 1:
                        s['final_row'] = s['filter_df'].iloc[0]
                        s['step'] = 'final_recommendation'
                        return self._final_recommendation()
                else:
                    print(f"[DEBUG] No current_followup_col found")
                    print(f"[DEBUG] Available follow-up state: {[k for k, v in s.items() if 'followup' in k or 'current' in k]}")
                # Find the next unanswered follow-up question (ONLY 1-3)
                found_next = False
                for idx, row in s['filter_df'].iterrows():
                    for i in range(1, 4):
                        q_col = f"Follow up question {i}"
                        a_col = f"Answer {i}"
                        fq = row.get(q_col, None) if hasattr(row, 'get') else None
                        if fq is None or pd.isna(fq) or not str(fq).strip():
                            continue
                        q = fq
                        if q and f"followup_{i}_answered" not in s:
                            s[f"current_followup_col"] = a_col
                            s[f"current_followup_qcol"] = q_col
                            s[f"current_row_index"] = idx
                            norm_q = normalize_question(q)
                            for d in (FOLLOWUP_1_QUESTIONS, FOLLOWUP_2_QUESTIONS, FOLLOWUP_3_QUESTIONS):
                                for key in d:
                                    if normalize_question(key) == norm_q:
                                        return {"response": q, "options": d[key]}
                            return {"response": q, "options": ["Yes", "No"]}
                # If no more follow-ups, always give the recommendation from the first row
                if not found_next:
                    s['current_followup_col'] = None
                    s['current_followup_qcol'] = None
                    s['current_row_index'] = None
                    if s['filter_df'] is not None and len(s['filter_df']) > 0:
                        s['final_row'] = s['filter_df'].iloc[0]
                        s['step'] = 'final_recommendation'
                        return self._final_recommendation()
                    else:
                        s['step'] = 'done'
                        return "Sorry, no matching profile found. Please consult a doctor."
                return self._followup_prompt()
        if s['step'] == 'final_recommendation':
            # After recommendation, expect medicine check
            s['step'] = 'medicine_check'
            return self._final_recommendation()
        if s.get('step') == 'medicine_check':
            return self._medicine_check(user_input)
        if s.get('step') == 'dispense_prompt':
            # Handle dispensing toggles
            return self._dispense_options(user_input)
        if s.get('step') == 'dispense_confirm':
            # Confirm dispensing selection
            selection = s.get('dispense_selection', None)
            if selection:
                s['step'] = 'done'
                return f"Thank you for selecting {selection}. Your selection has been recorded."
            else:
                s['step'] = 'done'
                return "Thank you. Your selection has been recorded."
        if s['step'] == 'done':
            return "Sorry, no matching profile found. Please consult a doctor."
        return "Sorry, I didn't understand."

    def _filter_and_followup(self):
        """
        Filter the dataset based on all collected symptoms and demographics, then proceed to follow-up questions.
        """
        s = self.state
        filter_df = self.df.copy()
        before_count = len(filter_df)
        filter_df = filter_df[
            (filter_df["Gender"] == s['gender']) &
            (filter_df["Age"].isin(s['age_bucket'])) &
            (filter_df["Weight"] == s['weight_bucket'])
        ]
        after_count = len(filter_df)
        if filter_df.empty:
            s['step'] = 'done'
            return "Sorry, no matching profile found. Please consult a doctor."
        for sym, val in s['user_flags'].items():
            before_count = len(filter_df)
            condition = filter_df[sym] == val
            matching_rows = filter_df[condition]
            filter_df = matching_rows
            after_count = len(filter_df)
            if filter_df.empty:
                s['step'] = 'done'
                return "Sorry, no matching profile found. Please consult a doctor."
        print(f"[DEBUG] Final filtered dataset shape: {filter_df.shape}")
        s['filter_df'] = filter_df
        if len(filter_df) == 1:
            s['final_row'] = filter_df.iloc[0]
            s['step'] = 'final_recommendation'
            return self._final_recommendation()
        s['step'] = 'followup_loop'
        s['followup_index'] = 1
        # If we're recomputing after a follow-up answer, we need to apply the follow-up filters
        if any(f"followup_{i}_answer" in s for i in range(1, 4)):
            for i in range(1, 4):  # Only 1-3
                ans_key = f"followup_{i}_answer"
                if ans_key in s:
                    ans = s[ans_key]
                    col = f"Answer {i}"
                    if col in filter_df.columns:
                        filter_df = filter_df[filter_df[col].astype(str).str.strip().str.lower() == ans.strip().lower()]
            s['filter_df'] = filter_df
            if len(filter_df) == 1:
                s['final_row'] = filter_df.iloc[0]
                s['step'] = 'final_recommendation'
                return self._final_recommendation()
        # If we have a pending follow-up answer, process it now
        if '_pending_followup_answer' in s:
            col, ans = s.pop('_pending_followup_answer')
            if col not in s['filter_df'].columns or not any(col == f"Answer {i}" for i in range(1, 4)):
                s['current_followup_col'] = None
                s['current_followup_qcol'] = None
                s['current_row_index'] = None
                # After skipping, check for more follow-ups or finish
                found_next = False
                for idx, row in s['filter_df'].iterrows():
                    for i in range(1, 4):
                        q_col = f"Follow up question {i}"
                        a_col = f"Answer {i}"
                        if q_col in row and pd.notna(row[q_col]) and str(row[q_col]).strip() != "" and f"followup_{i}_answered" not in s:
                            s[f"current_followup_col"] = a_col
                            s[f"current_followup_qcol"] = q_col
                            s[f"current_row_index"] = idx
                            found_next = True
                            return str(row[q_col]).strip()
                # If no more follow-ups, recommend or fallback
                if not found_next and s['filter_df'] is not None and len(s['filter_df']) >= 1:
                    s['final_row'] = s['filter_df'].iloc[0]
                    s['step'] = 'final_recommendation'
                    return self._final_recommendation()
                else:
                    s['step'] = 'done'
                    return "Sorry, no matching profile found. Please consult a doctor."
            unique_values = s['filter_df'][col].astype(str).str.strip().str.lower().unique()
            ans_lower = ans.strip().lower()
            matching_rows = s['filter_df'][s['filter_df'][col].astype(str).str.strip().str.lower() == ans_lower]
            s[f"followup_{col.split()[-1]}_answer"] = ans_lower
            s[f"followup_{col.split()[-1]}_answered"] = True
            s['current_followup_col'] = None
            s['current_followup_qcol'] = None
            s['current_row_index'] = None
            s['filter_df'] = matching_rows
            if s['filter_df'].empty:
                s['step'] = 'done'
                return "Sorry, no matching profile found. Please consult a doctor."
            if len(s['filter_df']) == 1:
                s['final_row'] = s['filter_df'].iloc[0]
                s['step'] = 'final_recommendation'
                return self._final_recommendation()
        return self._followup_prompt()

    def _ml_symptom_prompt(self):
        s = self.state
        # Find next unasked symptom
        unasked = [sym for sym in s['symptom_list'] if sym not in s['asked']]
        # Prepare demographics dict if available
        demographics = {}
        if s.get('age_bucket') is not None:
            demographics['Age'] = s['age_bucket']
        if s.get('weight_bucket') is not None:
            demographics['Weight'] = s['weight_bucket']
        if s.get('gender') is not None:
            demographics['Gender'] = s['gender']
        ranked = self.predictor.get_next_best_symptom(s['user_flags'], unasked, demographics)
        ranked = [r for r in ranked if r['probability'] >= 0.1]
        if not ranked:
            # No more symptoms to ask, filter the dataset
            s['step'] = 'filter_and_followup'
            return self._filter_and_followup()
        s['ml_symptom'] = ranked[0]['symptom']
        return {"response": f"Do you also have {s['ml_symptom']}? (Yes/No)", "options": ["Yes", "No"]}

    def _followup_prompt(self):
        s = self.state
        if s['filter_df'] is None or s['filter_df'].empty:
            s['step'] = 'done'
            return "Sorry, no matching profile found. Please consult a doctor."
        # Find the next unanswered follow-up question from any row in filter_df (ONLY 1-3)
        for idx, row in s['filter_df'].iterrows():
            for i in range(1, 4):
                q_col = f"Follow up question {i}"
                a_col = f"Answer {i}"
                if q_col in row and pd.notna(row[q_col]) and str(row[q_col]).strip() != "":
                    q = str(row[q_col]).strip()
                    if q and f"followup_{i}_answered" not in s:
                        s[f"current_followup_col"] = a_col
                        s[f"current_followup_qcol"] = q_col
                        s[f"current_row_index"] = idx
                        norm_q = normalize_question(q)
                        for d in (FOLLOWUP_1_QUESTIONS, FOLLOWUP_2_QUESTIONS, FOLLOWUP_3_QUESTIONS):
                            for key in d:
                                if normalize_question(key) == norm_q:
                                    return {"response": q, "options": d[key]}
                        return {"response": q, "options": ["Yes", "No"]}
        # If no more follow-ups found, proceed to recommendation
        if s['filter_df'] is not None and len(s['filter_df']) >= 1:
            s['final_row'] = s['filter_df'].iloc[0]
            s['step'] = 'final_recommendation'
            return self._final_recommendation()
        else:
            s['step'] = 'done'
            return "Sorry, no matching profile found. Please consult a doctor."

    def _final_recommendation(self):
        s = self.state
        row = s['final_row']
        rec = row["OTC/Doc"] if "OTC/Doc" in row else row["Otc/Doc"]
        
        # 🔄 NEW: Add complete conversation to learning system (only once per conversation)
        demographics = {}
        if s.get('age_bucket') is not None:
            demographics['Age'] = s['age_bucket']
        if s.get('weight_bucket') is not None:
            demographics['Weight'] = s['weight_bucket']
        if s.get('gender') is not None:
            demographics['Gender'] = s['gender']
        
        # Add complete conversation to learning system
        self.predictor.add_user_interaction(
            demographics=demographics,
            symptoms=s['user_flags'].copy(),
            final_recommendation=rec,
            success_rating=None  # Could be added later with user feedback
        )
        
        print(f"🔄 Learning: Added complete conversation to training data (recommendation: '{rec}')")
        # If recommendation is consult doctor, stop after showing recommendation
        if 'consult doctor' in rec.lower():
            s['step'] = 'done'
            return f"Based on your profile and symptoms:\n→ Recommendation: {rec}"
        # More precise medicine matching (case-insensitive, punctuation-stripped)
        def strip_punct(word):
            return word.translate(str.maketrans('', '', string.punctuation))
        matched_medicines = []
        rec_lower = rec.lower()
        # Split and strip punctuation from each word
        rec_words = set(strip_punct(w) for w in rec_lower.split())
        for med in MEDICINE_LIST:
            med_norm = med.strip().lower()
            med_words = set(strip_punct(w) for w in med_norm.split())
            # Check for exact word matches - all medicine words must be present as complete words
            if med_words.issubset(rec_words):
                if not any(m.strip().lower() == med_norm for m in matched_medicines):
                    matched_medicines.append(med)
            # For single-word medicines, check if it's a complete word match
            elif len(med_words) == 1:
                med_word = list(med_words)[0]
                if med_word in rec_words:
                    if not any(m.strip().lower() == med_norm for m in matched_medicines):
                        matched_medicines.append(med)
            # Handle medicines with parentheses and special characters
            else:
                # Remove parentheses and special characters for comparison
                med_clean = re.sub(r'[()]', '', med_norm)
                rec_clean = re.sub(r'[()]', '', rec_lower)
                if med_clean in rec_clean:
                    if not any(m.strip().lower() == med_norm for m in matched_medicines):
                        matched_medicines.append(med)
        if matched_medicines:
            filtered_meds = filter_longest_medicines(matched_medicines)
            med_str = ', '.join(filtered_meds)
            s['recommended_medicine'] = med_str
            # Set step to medicine_check for next user input
            s['step'] = 'medicine_check'
            return f"Based on your profile and symptoms:\n→ Recommendation: {rec}\n\nHave you taken any other medicine during these days? If yes, please tell me which medicine. If not, type 'no'."
        else:
            s['recommended_medicine'] = ''
            s['step'] = 'done'
            return f"Based on your profile and symptoms:\n→ Recommendation: {rec}"

    def _medicine_check(self, user_input):
        s = self.state
        recommended = s.get('recommended_medicine', '').lower()
        user_input = user_input.strip().lower()
        # Acceptable 'no' responses
        no_responses = {'no', 'none', 'nothing', 'not taken', 'did not take', 'haven\'t taken', 'haven\'t', 'didn\'t', 'nil'}
        # If user says no
        if any(word == user_input for word in no_responses):
            s['step'] = 'dispense_prompt'
            s.pop('dispense_options', None)
            return self._dispense_options('')
        # Split user input by common delimiters to handle multiple medicines
        user_meds = re.split(r'[,&;]| and | with | plus ', user_input)
        user_meds = [m.strip() for m in user_meds if m.strip()]
        # Check if any user medicine matches recommended
        rec_meds = [m.strip().lower() for m in s.get('recommended_medicine', '').split(',') if m.strip()]
        matched = False
        for med in user_meds:
            for rec in rec_meds:
                if self.matches_recommended_medicine(med, rec):
                    matched = True
                    break
            if matched:
                break
        if matched:
            s['step'] = 'dispense_prompt'
            s.pop('dispense_options', None)
            return self._dispense_options('')
        else:
            s['step'] = 'done'
            return "It is better that you consult the doctor since you've already taken a different medicine."

    def matches_recommended_medicine(self, user_input, recommended_medicine):
        # Normalize
        user_input = user_input.lower().strip()
        recommended_medicine = recommended_medicine.lower().strip()
        # Remove dosage for comparison
        user_base = re.sub(r'\b\d+\s*mg\b', '', user_input)
        rec_base = re.sub(r'\b\d+\s*mg\b', '', recommended_medicine)
        user_base = re.sub(r'[^a-z0-9 ]', '', user_base)
        rec_base = re.sub(r'[^a-z0-9 ]', '', rec_base)
        # Fuzzy match threshold
        threshold = 0.7
        # Direct substring or fuzzy match
        if user_base in rec_base or rec_base in user_base:
            return True
        # Brand-generic mappings
        MEDICINE_MAPPINGS = {
            'paracetamol': ['crocin', 'dolo', 'tylenol', 'fever medicine'],
            'cetirizine': ['cetrizine', 'zyrtec', 'allegra-like'],
            'pantoprazole': ['acidity medicine', 'proton pump inhibitor'],
            'gelusil': ['antacid', 'acidity tablet'],
        }
        # Check mapping
        for generic, brands in MEDICINE_MAPPINGS.items():
            if generic in rec_base:
                for b in brands + [generic]:
                    if b in user_base or user_base in b:
                        return True
        # Fuzzy match
        ratio = difflib.SequenceMatcher(None, user_base, rec_base).ratio()
        if ratio >= threshold:
            return True
        return False

    def get_greeting(self):
        return "Welcome to SmartMed AI Terminal.\nWhat is your age (in years)?"

    def get_serializable_state(self):
        serializable = self.state.copy()
        # Convert all sets to lists for serialization
        for k, v in serializable.items():
            if isinstance(v, set):
                serializable[k] = list(v)
        # Remove or nullify DataFrames and Series (not serializable)
        if 'filter_df' in serializable and isinstance(serializable['filter_df'], pd.DataFrame):
            serializable['filter_df'] = None
        if 'final_row' in serializable and isinstance(serializable['final_row'], pd.Series):
            serializable['final_row'] = None
        return serializable

    def restore_state(self, state):
        self.state = state
        # Convert all list fields that should be sets back to sets
        for k, v in self.state.items():
            if k == 'asked' and isinstance(v, list):
                self.state[k] = set(v)

    def bucket_age(self, age: int) -> list:
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

    def _dispense_options(self, user_input):
        s = self.state
        # If this is the first entry to this step, present toggles
        if 'dispense_options' not in s:
            # Parse recommended medicines
            rec_meds = [m.strip() for m in s.get('recommended_medicine', '').split(',') if m.strip()]
            options = []
            if len(rec_meds) == 1:
                options = [rec_meds[0], "No, I don't want to dispense any medicine"]
            elif len(rec_meds) == 2:
                a, b = rec_meds
                options = [f"{a} only", f"{b} only", f"Both {a} and {b}", "No, I don't want to dispense any medicine"]
            elif len(rec_meds) >= 3:
                from itertools import combinations
                meds = rec_meds
                # Individual
                for m in meds:
                    options.append(f"{m} only")
                # All pairs
                for comb in combinations(meds, 2):
                    options.append(f"{' and '.join(comb)}")
                # All
                options.append(f"All {' and '.join(meds)}" if len(meds) == 3 else f"All {' and '.join(meds)}")
                options.append("No, I don't want to dispense any medicine")
            s['dispense_options'] = options
            s['step'] = 'dispense_prompt'
            return {"response": "Do you want to dispense the medicine?", "options": options, "type": "medicine"}
        # Otherwise, process user selection
        selection = user_input.strip()
        valid = False
        # Accept both the 'no' button and 'n' as valid for 'no'
        for opt in s['dispense_options']:
            if selection.lower() == opt.lower() or (selection.lower() == 'n' and "no, i don't want to dispense any medicine" in opt.lower()):
                s['dispense_selection'] = opt
                valid = True
                break
        if not valid:
            # Try to match by index if user enters a number
            try:
                idx = int(selection) - 1
                if 0 <= idx < len(s['dispense_options']):
                    s['dispense_selection'] = s['dispense_options'][idx]
                    valid = True
            except:
                pass
        if valid:
            # If user selects 'No', print 'n' and finish
            if s['dispense_selection'].lower().startswith('no, i don') or selection.lower() == 'n':
                print('n')
                s['step'] = 'done'
                return 'OK, thank you. Take care.'
            else:
                # Check medicine availability before redirecting to payment
                med_names = []
                selection = s['dispense_selection']
                if selection.lower().startswith('both '):
                    med_names = [m.strip() for m in selection[5:].split(' and ')]
                elif selection.lower().startswith('all '):
                    med_names = [m.strip() for m in selection[4:].split(' and ')]
                elif selection.lower().endswith(' only'):
                    med_names = [selection[:-5].strip()]
                else:
                    med_names = [selection.strip()]
                
                # Check if any medicine is unavailable (this will be verified again in medicine_gateway)
                from flask import current_app
                db = current_app.config['DATABASE']
                unavailable_meds = []
                for med in med_names:
                    medicine_doc = db.medicines.find_one({'name': med})
                    if not medicine_doc or medicine_doc.get('quantity', 0) <= 0:
                        unavailable_meds.append(med)
                
                if unavailable_meds:
                    # Return error message if any medicine is unavailable
                    error_msg = f"Medicine not available: {', '.join(unavailable_meds)}"
                    return {"response": error_msg, "type": "error"}
                else:
                    # Redirect to medicine gateway for payment
                    return {'redirect': f"/medicine_gateway?medicine={s['dispense_selection']}"}
        else:
            return {"response": "Please select a valid option:", "options": s['dispense_options'], "type": "medicine"}