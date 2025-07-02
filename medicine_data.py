from typing import List, Dict

MEDICINE_LIST = {
    'pain_relief': [
        {'name': 'Aspirin', 'class': 'NSAID', 'conditions': ['pain', 'fever', 'inflammation']},
        {'name': 'Ibuprofen', 'class': 'NSAID', 'conditions': ['pain', 'fever', 'inflammation']},
        {'name': 'Naproxen', 'class': 'NSAID', 'conditions': ['pain', 'inflammation']},
        {'name': 'Acetaminophen', 'class': 'Analgesic', 'conditions': ['pain', 'fever']},
        {'name': 'Celecoxib', 'class': 'COX-2 Inhibitor', 'conditions': ['arthritis', 'chronic pain']},
    ],
    'gastrointestinal': [
        {'name': 'Antacids', 'class': 'Acid Neutralizer', 'conditions': ['heartburn', 'indigestion']},
        {'name': 'Omeprazole', 'class': 'Proton Pump Inhibitor', 'conditions': ['acid reflux', 'ulcers']},
        {'name': 'Ranitidine', 'class': 'H2 Blocker', 'conditions': ['heartburn', 'acid reflux']},
        {'name': 'Simethicone', 'class': 'Anti-gas', 'conditions': ['bloating', 'gas']},
    ],
    'migraine': [
        {'name': 'Sumatriptan', 'class': 'Triptan', 'conditions': ['migraine']},
        {'name': 'Rizatriptan', 'class': 'Triptan', 'conditions': ['migraine']},
        {'name': 'Excedrin Migraine', 'class': 'Combination', 'conditions': ['migraine', 'headache']},
    ],
    'allergy': [
        {'name': 'Diphenhydramine', 'class': 'Antihistamine', 'conditions': ['allergies', 'itching']},
        {'name': 'Loratadine', 'class': 'Antihistamine', 'conditions': ['allergies', 'hay fever']},
        {'name': 'Cetirizine', 'class': 'Antihistamine', 'conditions': ['allergies', 'hives']},
    ],
    'cold_and_flu': [
        {'name': 'Dextromethorphan', 'class': 'Cough Suppressant', 'conditions': ['cough']},
        {'name': 'Guaifenesin', 'class': 'Expectorant', 'conditions': ['chest congestion']},
        {'name': 'Phenylephrine', 'class': 'Decongestant', 'conditions': ['nasal congestion']},
    ],
    'first_aid': [
        {'name': 'Silver sulfadiazine', 'class': 'Topical Antibiotic', 'conditions': ['burns', 'skin infections']},
        {'name': 'Povidone-Iodine', 'class': 'Antiseptic', 'conditions': ['cuts', 'scrapes']},
        {'name': 'Hydrocortisone cream', 'class': 'Corticosteroid', 'conditions': ['itching', 'rash']},
    ]
}

def get_medicine_by_symptoms(symptoms: List[str]) -> List[Dict]:
    """Get appropriate medicines based on symptoms."""
    matches = []
    for category, medicines in MEDICINE_LIST.items():
        for medicine in medicines:
            if any(symptom.lower() in [cond.lower() for cond in medicine['conditions']] for symptom in symptoms):
                matches.append(medicine)
    return matches

def get_medicine_categories() -> List[str]:
    """Get all available medicine categories."""
    return list(MEDICINE_LIST.keys())

def get_medicines_in_category(category: str) -> List[Dict]:
    """Get all medicines in a specific category."""
    return MEDICINE_LIST.get(category, [])

def is_serious_condition(symptoms: List[str]) -> bool:
    """Determine if symptoms indicate a serious condition requiring medical attention."""
    serious_symptoms = [
        'chest pain', 'difficulty breathing', 'severe pain',
        'unconscious', 'seizure', 'stroke', 'heart attack',
        'severe bleeding', 'head injury', 'poisoning',
        'severe allergic reaction', 'anaphylaxis'
    ]
    return any(serious in ' '.join(symptoms).lower() for serious in serious_symptoms)