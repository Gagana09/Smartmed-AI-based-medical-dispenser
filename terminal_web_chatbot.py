import pandas as pd
from symptom_predictor import SymptomPredictor, SYMPTOMS
import threading

DATASET_PATH = r"C:\Users\achar\OneDrive\Desktop\IDP\new_idp\Smartmed-AI-based-medical-dispenser\dataset_1.xlsx"
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
            _GLOBAL_PREDICTOR = SymptomPredictor()
            _GLOBAL_PREDICTOR.train_on_csv(DATASET_PATH)
    return _GLOBAL_DF, _GLOBAL_PREDICTOR

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
        print("\n[DEBUG] ====== ENTER get_response ======")
        print("[DEBUG] State at start:", s)
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
            prob = ranked[0]['probability']
            return {
                "response": f"Do you also have {s['ml_symptom']}? (Yes/No) [Confidence: {prob*100:.1f}%]",
                "options": ["Yes", "No"]
            }
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
                s['ml_symptom'] = None  # Clear the current symptom
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
            prob = ranked[0]['probability']
            return {
                "response": f"Do you also have {s['ml_symptom']}? (Yes/No) [Confidence: {prob*100:.1f}%]",
                "options": ["Yes", "No"]
            }
        if s['step'] == 'filter_and_followup':
            return self._filter_and_followup()
        if s['step'] == 'followup_loop':
            print(f"[DEBUG] Followup loop - user input: '{user_input}'")
            print(f"[DEBUG] Current followup col: {s.get('current_followup_col')}")
            print(f"[DEBUG] Filter_df is None: {s['filter_df'] is None}")
            print(f"[DEBUG] Filter_df shape: {s['filter_df'].shape if s['filter_df'] is not None else 'None'}")
            if s['filter_df'] is None:
                print("[DEBUG] filter_df is None in followup_loop, recomputing with _filter_and_followup()!")
                s['step'] = 'filter_and_followup'
                # Save the pending follow-up answer to process after recompute
                if s.get('current_followup_col'):
                    s['_pending_followup_answer'] = (s['current_followup_col'], user_input.strip())
                return self._filter_and_followup()
            if s['filter_df'] is not None:
                print(f"[DEBUG] Filter_df columns: {list(s['filter_df'].columns)}")
            if s.get('current_followup_col'):
                print(f"[DEBUG] Found current_followup_col: {s.get('current_followup_col')}")
                ans = user_input.strip()
                col = s['current_followup_col']
                if col not in s['filter_df'].columns:
                    print(f"[DEBUG] Column {col} not in DataFrame, skipping. Columns are: {list(s['filter_df'].columns)}")
                    s['current_followup_col'] = None
                    s['current_followup_qcol'] = None
                    s['current_row_index'] = None
                    for idx, row in s['filter_df'].iterrows():
                        for i in range(1, 5):
                            q_col = f"Follow up question {i}"
                            a_col = f"Answer {i}"
                            if q_col in row and pd.notna(row[q_col]) and str(row[q_col]).strip() != "" and f"followup_{i}_answered" not in s:
                                s[f"current_followup_col"] = a_col
                                s[f"current_followup_qcol"] = q_col
                                s[f"current_row_index"] = idx
                                print(f"[DEBUG] Next follow-up after skip: {q_col} -> {str(row[q_col]).strip()}")
                                return str(row[q_col]).strip()
                    if s['filter_df'] is not None and len(s['filter_df']) >= 1:
                        print(f"[DEBUG] No more follow-ups, giving recommendation.")
                        s['final_row'] = s['filter_df'].iloc[0]
                        s['step'] = 'final_recommendation'
                        return self._final_recommendation()
                    else:
                        print(f"[DEBUG] No more follow-ups and no rows left, fallback.")
                        s['step'] = 'done'
                        return "Sorry, no matching profile found. Please consult a doctor."
                print(f"[DEBUG] Processing follow-up answer: {col} = {ans}")
                unique_values = s['filter_df'][col].astype(str).str.strip().str.lower().unique()
                print(f"[DEBUG] Available values for {col}: {unique_values}")
                ans_lower = ans.strip().lower()
                matching_rows = s['filter_df'][s['filter_df'][col].astype(str).str.strip().str.lower() == ans_lower]
                print(f"[DEBUG] Found {len(matching_rows)} rows matching {col}={ans}")
                if len(matching_rows) == 0:
                    print(f"[DEBUG] No exact match, trying partial matching")
                    for val in unique_values:
                        if ans_lower in val or val in ans_lower:
                            print(f"[DEBUG] Partial match found: '{ans_lower}' matches '{val}'")
                            ans_lower = val
                            matching_rows = s['filter_df'][s['filter_df'][col].astype(str).str.strip().str.lower() == ans_lower]
                            break
                s[f"followup_{col.split()[-1]}_answer"] = ans_lower
                s[f"followup_{col.split()[-1]}_answered"] = True
                s['current_followup_col'] = None
                s['current_followup_qcol'] = None
                s['current_row_index'] = None
                s['filter_df'] = matching_rows
                print(f"[DEBUG] After follow-up filtering: {len(s['filter_df'])} rows")
                if s['filter_df'].empty:
                    print(f"[DEBUG] All rows filtered out after follow-up answer.")
                    s['step'] = 'done'
                    return "Sorry, no matching profile found. Please consult a doctor."
                if len(s['filter_df']) == 1:
                    print(f"[DEBUG] Only one row left after follow-up, giving recommendation.")
                    s['final_row'] = s['filter_df'].iloc[0]
                    s['step'] = 'final_recommendation'
                    return self._final_recommendation()
            else:
                print(f"[DEBUG] No current_followup_col found")
                print(f"[DEBUG] Available follow-up state: {[k for k, v in s.items() if 'followup' in k or 'current' in k]}")
            # Find the next unanswered follow-up question (ONLY 1-3)
            found_next = False
            for idx, row in s['filter_df'].iterrows():
                for i in range(1, 4):  # Only 1-3
                    q_col = f"Follow up question {i}"
                    a_col = f"Answer {i}"
                    fq = row.get(q_col, None) if hasattr(row, 'get') else None
                    if fq is None or pd.isna(fq) or not str(fq).strip():
                        continue
                    q = fq
                    if q and f"followup_{i}_answered" not in s:
                        if a_col not in s['filter_df'].columns:
                            print(f"[DEBUG] {a_col} not in DataFrame, giving recommendation from first row. Columns are: {list(s['filter_df'].columns)}")
                            s['final_row'] = s['filter_df'].iloc[0]
                            s['step'] = 'final_recommendation'
                            # Clear follow-up state
                            s['current_followup_col'] = None
                            s['current_followup_qcol'] = None
                            s['current_row_index'] = None
                            return self._final_recommendation()
                        s[f"current_followup_col"] = a_col
                        s[f"current_followup_qcol"] = q_col
                        s[f"current_row_index"] = idx
                        print(f"[DEBUG] Next follow-up: {q_col} -> {q}")
                        found_next = True
                        return q
            # If no more follow-ups, always give the recommendation from the first row
            if not found_next:
                s['current_followup_col'] = None
                s['current_followup_qcol'] = None
                s['current_row_index'] = None
                if s['filter_df'] is not None and len(s['filter_df']) > 0:
                    print(f"[DEBUG] All follow-ups answered or no more valid follow-ups. Giving top recommendation.")
                    s['final_row'] = s['filter_df'].iloc[0]
                    s['step'] = 'final_recommendation'
                    return self._final_recommendation()
                else:
                    print(f"[DEBUG] No more follow-ups and no rows left, fallback.")
                    s['step'] = 'done'
                    return "Sorry, no matching profile found. Please consult a doctor."
            return self._followup_prompt()
        if s['step'] == 'final_recommendation':
            return self._final_recommendation()
        if s['step'] == 'done':
            return "Sorry, no matching profile found. Please consult a doctor."
        print("[DEBUG] State at end:", s)
        return "Sorry, I didn't understand."

    def _filter_and_followup(self):
        """
        Filter the dataset based on all collected symptoms and demographics, then proceed to follow-up questions.
        """
        s = self.state
        print(f"[DEBUG] Current symptoms: {s['user_flags']}")
        print(f"[DEBUG] Dataset shape before filtering: {self.df.shape}")
        filter_df = self.df.copy()
        before_count = len(filter_df)
        filter_df = filter_df[
            (filter_df["Gender"] == s['gender']) &
            (filter_df["Age"].isin(s['age_bucket'])) &
            (filter_df["Weight"] == s['weight_bucket'])
        ]
        after_count = len(filter_df)
        print(f"[DEBUG] After demographics filter: {before_count} -> {after_count} rows")
        if filter_df.empty:
            s['step'] = 'done'
            return "Sorry, no matching profile found. Please consult a doctor."
        for sym, val in s['user_flags'].items():
            before_count = len(filter_df)
            condition = filter_df[sym] == val
            matching_rows = filter_df[condition]
            print(f"[DEBUG] Looking for {sym}={val}")
            print(f"[DEBUG] Found {len(matching_rows)} rows with {sym}={val}")
            if len(matching_rows) > 0:
                print(f"[DEBUG] Sample rows with {sym}={val}:")
                print(matching_rows[['Age', 'Gender', 'Weight', sym]][:3])
            filter_df = matching_rows
            after_count = len(filter_df)
            print(f"[DEBUG] Filtering {sym}={val}: {before_count} -> {after_count} rows")
            if filter_df.empty:
                print(f"[DEBUG] No rows left after filtering {sym}={val}")
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
            print("[DEBUG] Applying follow-up filters after recomputation")
            for i in range(1, 4):  # Only 1-3
                ans_key = f"followup_{i}_answer"
                if ans_key in s:
                    ans = s[ans_key]
                    col = f"Answer {i}"
                    if col in filter_df.columns:
                        before_count = len(filter_df)
                        filter_df = filter_df[filter_df[col].astype(str).str.strip().str.lower() == ans.strip().lower()]
                        after_count = len(filter_df)
                        print(f"[DEBUG] Follow-up filter {col}={ans}: {before_count} -> {after_count} rows")
                        if filter_df.empty:
                            s['step'] = 'done'
                            return "Sorry, no matching profile found. Please consult a doctor."
            s['filter_df'] = filter_df
            if len(filter_df) == 1:
                s['final_row'] = filter_df.iloc[0]
                s['step'] = 'final_recommendation'
                return self._final_recommendation()
        # If we have a pending follow-up answer, process it now
        if '_pending_followup_answer' in s:
            col, ans = s.pop('_pending_followup_answer')
            print(f"[DEBUG] Processing pending follow-up answer after recompute: {col} = {ans}")
            # PATCH: Only process if col exists and is Answer 1-3
            if col not in s['filter_df'].columns or not any(col == f"Answer {i}" for i in range(1, 4)):
                print(f"[DEBUG] Column {col} not in DataFrame or not a valid follow-up, skipping.")
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
                            print(f"[DEBUG] Next follow-up after skip: {q_col} -> {str(row[q_col]).strip()}")
                            found_next = True
                            return str(row[q_col]).strip()
                # If no more follow-ups, recommend or fallback
                if not found_next and s['filter_df'] is not None and len(s['filter_df']) >= 1:
                    print(f"[DEBUG] No more follow-ups, giving recommendation.")
                    s['final_row'] = s['filter_df'].iloc[0]
                    s['step'] = 'final_recommendation'
                    return self._final_recommendation()
                else:
                    print(f"[DEBUG] No more follow-ups and no rows left, fallback.")
                    s['step'] = 'done'
                    return "Sorry, no matching profile found. Please consult a doctor."
            unique_values = s['filter_df'][col].astype(str).str.strip().str.lower().unique()
            print(f"[DEBUG] Available values for {col}: {unique_values}")
            ans_lower = ans.strip().lower()
            matching_rows = s['filter_df'][s['filter_df'][col].astype(str).str.strip().str.lower() == ans_lower]
            print(f"[DEBUG] Found {len(matching_rows)} rows matching {col}={ans}")
            if len(matching_rows) == 0:
                print(f"[DEBUG] No exact match, trying partial matching")
                for val in unique_values:
                    if ans_lower in val or val in ans_lower:
                        print(f"[DEBUG] Partial match found: '{ans_lower}' matches '{val}'")
                        ans_lower = val
                        matching_rows = s['filter_df'][s['filter_df'][col].astype(str).str.strip().str.lower() == ans_lower]
                        break
            s[f"followup_{col.split()[-1]}_answer"] = ans_lower
            s[f"followup_{col.split()[-1]}_answered"] = True
            s['current_followup_col'] = None
            s['current_followup_qcol'] = None
            s['current_row_index'] = None
            s['filter_df'] = matching_rows
            print(f"[DEBUG] After follow-up filtering: {len(s['filter_df'])} rows")
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
        prob = ranked[0]['probability']
        return f"Do you also have {s['ml_symptom']}? (Yes/No) [Confidence: {prob*100:.1f}%]"

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
                        return q
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
        s['step'] = 'done'

        # Partial matching for all medicines (case-insensitive)
        matched_medicines = []
        rec_lower = rec.lower()
        for med in MEDICINE_LIST:
            med_norm = med.strip().lower()
            if any(part.strip().lower() in rec_lower for part in med.lower().split(',')):
                # Avoid duplicates (case-insensitive)
                if not any(m.strip().lower() == med_norm for m in matched_medicines):
                    matched_medicines.append(med)
        # Filter out medicines that are substrings of longer ones (case-insensitive)
        if matched_medicines:
            filtered_meds = filter_longest_medicines(matched_medicines)
            med_str = ', '.join(filtered_meds)
            rec += f"\n\nDo you want to dispense {med_str}?"
        return f"Based on your profile and symptoms:\n→ Recommendation: {rec}"

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