import pandas as pd
import numpy as np
import re
import warnings
from typing import List, Dict, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
import subprocess
import sys
import os

warnings.filterwarnings('ignore')

class MedicalNLPPreprocessor:
    def __init__(self, use_advanced_models: bool = True):
        """Initialize with medical models (with fallback options)"""
        print("Initializing Medical NLP Preprocessor...")
        
        self.use_advanced_models = use_advanced_models
        self.medical_nlp = None
        self.clinical_tokenizer = None
        self.clinical_model = None
        self.sentence_model = None
        self.medical_ner = None
        
        # Try to load advanced models if requested
        if use_advanced_models:
            self._load_advanced_models()
        
        # Initialize symptom knowledge base
        self.symptom_embeddings = None
        self.symptom_labels = None
        self._build_symptom_knowledge_base()
        
    def _safe_install(self, package: str, model_name: str = None) -> bool:
        """Safely install packages with error handling"""
        import sys
        import subprocess
        
        # Check if we're in a virtual environment
        in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
        
        try:
            if model_name:
                print(f"Installing {model_name}...")
            else:
                print(f"Installing {package}...")
            
            #Use --user only if NOT in virtual environment
            install_cmd = [sys.executable, "-m", "pip", "install", package, "--upgrade", "--no-deps"]
            if not in_venv:
                install_cmd.insert(-2, "--user")  # Insert before --upgrade
            
            subprocess.check_call(install_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except subprocess.CalledProcessError:
            print(f"Failed to install {package}")
            return False
        except Exception as e:
            print(f"Error installing {package}: {e}")
            return False
          
    def _load_advanced_models(self):
        """Load advanced medical models with comprehensive fallbacks"""
        
        # Try to load spaCy models
        try:
            import spacy
            # Try medical spaCy first
            try:
                self.medical_nlp = spacy.load("en_core_sci_sm")
                print("✓ SciSpaCy medical model loaded")
            except OSError:
                # Try to install and load
                if self._safe_install("scispacy", "SciSpaCy"):
                    if self._safe_install("https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_sm-0.5.3.tar.gz", "Medical model"):
                        try:
                            self.medical_nlp = spacy.load("en_core_sci_sm")
                            print("✓ SciSpaCy medical model installed and loaded")
                        except:
                            pass
                
                # Fallback to standard spaCy
                if not self.medical_nlp:
                    try:
                        self.medical_nlp = spacy.load("en_core_web_sm")
                        print("✓ Standard spaCy model loaded")
                    except OSError:
                        print("⚠ SpaCy models not available, using basic NER")
                        
        except ImportError:
            print("⚠ SpaCy not available")
        
        # Try to load transformers models
        try:
            from transformers import AutoTokenizer, AutoModel, pipeline
            import torch
            
            # Try ClinicalBERT
            try:
                self.clinical_tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
                self.clinical_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
                print("✓ ClinicalBERT loaded")
            except Exception:
                # Try PubMedBERT
                try:
                    self.clinical_tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
                    self.clinical_model = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
                    print("✓ PubMedBERT loaded")
                except Exception:
                    # Fallback to DistilBERT
                    try:
                        self.clinical_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
                        self.clinical_model = AutoModel.from_pretrained("distilbert-base-uncased")
                        print("✓ DistilBERT loaded as fallback")
                    except Exception:
                        print("⚠ No transformer models available")
                        
            # Try medical NER pipeline
            try:
                self.medical_ner = pipeline("ner", 
                                          model="d4data/biomedical-ner-all", 
                                          aggregation_strategy="simple",
                                          device=-1)
                print("✓ Medical NER pipeline loaded")
            except Exception:
                print("⚠ Medical NER not available")
                
        except ImportError:
            print("⚠ Transformers not available")
        
        # Try to load sentence transformers
        try:
            from sentence_transformers import SentenceTransformer
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✓ Sentence transformer loaded")
        except Exception:
            print("⚠ Sentence transformer not available")
    
    def _build_symptom_knowledge_base(self):
        """Build comprehensive symptom knowledge base"""
        standard_symptoms = [
            # Pain types
            "headache", "migraine", "tension headache", "cluster headache", "sinus headache",
            "body ache", "muscle pain", "joint pain", "back pain", "neck pain", "shoulder pain",
            "chest pain", "abdominal pain", "stomach pain", "pelvic pain",
            "arthritis pain", "fibromyalgia", "chronic pain", "acute pain", "neuropathic pain",
            
            # Fever and temperature
            "fever", "high fever", "low grade fever", "pyrexia", "high temperature", "chills",
            "night sweats", "hyperthermia",
            
            # Digestive issues
            "nausea", "vomiting", "diarrhea", "constipation", "heartburn", "acid reflux",
            "indigestion", "stomach upset", "bloating", "gas", "cramping",
            
            # Respiratory symptoms
            "cough", "dry cough", "productive cough", "shortness of breath", "wheezing",
            "chest congestion", "runny nose", "stuffy nose", "sore throat", "sneezing",
            
            # Skin conditions
            "rash", "itching", "hives", "eczema", "dry skin", "acne", "dermatitis",
            "burn", "cut", "wound", "bruise", "swelling", "inflammation",
            
            # Mental health
            "anxiety", "depression", "stress", "insomnia", "sleep problems", "fatigue",
            "mood swings", "irritability",
            
            # Specific conditions
            "allergy", "allergic reaction", "cold symptoms", "flu symptoms", "sinusitis",
            "urinary tract infection", "yeast infection", "menstrual cramps", "menopause symptoms",
            
            # Post-medical procedures
            "postoperative pain", "post-surgical pain", "dental pain", "tooth pain",
            "injection site pain", "vaccination side effects",
            
            # Injury-related
            "injury pain", "trauma pain", "sports injury", "sprain", "strain",
            "fracture pain", "dislocation", "burn pain", "sunburn"
        ]
        
        self.symptom_labels = standard_symptoms
        
        if self.sentence_model:
            try:
                self.symptom_embeddings = self.sentence_model.encode(standard_symptoms)
                print(f"✓ Symptom knowledge base built with embeddings ({len(standard_symptoms)} symptoms)")
            except Exception as e:
                print(f"⚠ Error creating embeddings: {e}")
                self.symptom_embeddings = None
        else:
            self.symptom_embeddings = None
            
        print(f"✓ Symptom knowledge base ready with {len(standard_symptoms)} symptoms")
    
    def extract_medical_entities_basic(self, text: str) -> Dict[str, List[str]]:
        """Basic medical entity extraction using pattern matching"""
        if pd.isna(text) or not text.strip():
            return {'DRUG': [], 'CONDITION': [], 'SYMPTOM': [], 'DOSAGE': []}
        
        text_lower = text.lower()
        entities = {'DRUG': [], 'CONDITION': [], 'SYMPTOM': [], 'DOSAGE': []}
        
        # Common drug patterns
        drug_patterns = [
            r'\b(?:acetaminophen|paracetamol|tylenol)\b',
            r'\b(?:ibuprofen|advil|motrin|brufen)\b',
            r'\b(?:aspirin|bayer|bufferin)\b',
            r'\b(?:naproxen|aleve|naprosyn)\b',
            r'\b(?:diclofenac|voltaren)\b',
            r'\b(?:celecoxib|celebrex)\b',
            r'\b(?:hydrocodone|vicodin)\b',
            r'\b(?:codeine|morphine|tramadol)\b',
            r'\b(?:prednisone|prednisolone)\b',
            r'\b(?:omeprazole|prilosec|nexium)\b',
            r'\b(?:loratadine|claritin|zyrtec|allegra)\b',
            r'\b(?:diphenhydramine|benadryl)\b',
            r'\b(?:pseudoephedrine|sudafed)\b',
            r'\b(?:dextromethorphan|robitussin)\b'
        ]
        
        for pattern in drug_patterns:
            matches = re.findall(pattern, text_lower)
            entities['DRUG'].extend(matches)
        
        # Common condition patterns
        condition_patterns = [
            r'\b(?:diabetes|hypertension|asthma|copd)\b',
            r'\b(?:heart disease|cardiac|cardiovascular)\b',
            r'\b(?:kidney disease|renal)\b',
            r'\b(?:liver disease|hepatic)\b',
            r'\b(?:pregnancy|pregnant|nursing|breastfeeding)\b',
            r'\b(?:allergy|allergic)\b',
            r'\b(?:epilepsy|seizure)\b'
        ]
        
        for pattern in condition_patterns:
            matches = re.findall(pattern, text_lower)
            entities['CONDITION'].extend(matches)
        
        # Dosage patterns
        dosage_patterns = [
            r'\b\d+\s*(?:mg|g|ml|tablet|capsule|dose)s?\b',
            r'\b(?:once|twice|three times|four times)\s+(?:daily|a day|per day)\b',
            r'\b(?:every|q)\s*\d+\s*(?:hours|hrs|h)\b'
        ]
        
        for pattern in dosage_patterns:
            matches = re.findall(pattern, text_lower)
            entities['DOSAGE'].extend(matches)
        
        return entities
    
    def extract_medical_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract medical entities using available models or fallback"""
        if pd.isna(text) or not text.strip():
            return {'DRUG': [], 'CONDITION': [], 'SYMPTOM': [], 'DOSAGE': []}
        
        entities = {'DRUG': [], 'CONDITION': [], 'SYMPTOM': [], 'DOSAGE': []}
        
        # Try advanced NER first
        if self.medical_ner:
            try:
                ner_results = self.medical_ner(text)
                for entity in ner_results:
                    label = entity['entity_group'].upper()
                    if 'DRUG' in label or 'CHEMICAL' in label:
                        entities['DRUG'].append(entity['word'])
                    elif 'DISEASE' in label or 'CONDITION' in label:
                        entities['CONDITION'].append(entity['word'])
                    elif 'SYMPTOM' in label or 'SIGN' in label:
                        entities['SYMPTOM'].append(entity['word'])
            except Exception as e:
                print(f"Advanced NER failed, using fallback: {e}")
        
        # Try spaCy NER
        if self.medical_nlp and not any(entities.values()):
            try:
                doc = self.medical_nlp(text)
                for ent in doc.ents:
                    if ent.label_ in ['DRUG', 'CHEMICAL']:
                        entities['DRUG'].append(ent.text.strip())
                    elif ent.label_ in ['DISEASE', 'CONDITION']:
                        entities['CONDITION'].append(ent.text.strip())
            except Exception as e:
                print(f"SpaCy NER failed: {e}")
        
        # Fallback to basic pattern matching
        if not any(entities.values()):
            basic_entities = self.extract_medical_entities_basic(text)
            for key in entities:
                entities[key].extend(basic_entities[key])
        
        # Remove duplicates and clean
        for key in entities:
            entities[key] = list(set([item.strip() for item in entities[key] if item.strip()]))
        
        return entities
    
    def get_clinical_embedding(self, text: str) -> np.ndarray:
        """Get clinical embeddings with fallback"""
        if pd.isna(text) or not text.strip():
            return np.zeros(768)
        
        if self.clinical_model and self.clinical_tokenizer:
            try:
                import torch
                inputs = self.clinical_tokenizer(
                    text, 
                    return_tensors="pt", 
                    truncation=True, 
                    padding=True, 
                    max_length=512
                )
                
                with torch.no_grad():
                    outputs = self.clinical_model(**inputs)
                    embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
                return embedding
            except Exception as e:
                print(f"Clinical embedding failed: {e}")
        
        # Fallback: simple text-based features
        return self._get_basic_text_features(text)
    
    def _get_basic_text_features(self, text: str) -> np.ndarray:
        """Create basic text features as embedding fallback"""
        if pd.isna(text) or not text.strip():
            return np.zeros(768)
        
        text_lower = text.lower()
        
        # Create basic features
        features = []
        
        # Length features
        features.extend([
            len(text),
            len(text.split()),
            len(re.findall(r'\w+', text))
        ])
        
        # Medical keyword features
        medical_keywords = [
            'pain', 'ache', 'fever', 'nausea', 'headache', 'cough', 'allergy',
            'infection', 'inflammation', 'chronic', 'acute', 'severe', 'mild',
            'tablet', 'capsule', 'mg', 'dose', 'daily', 'twice', 'prescription'
        ]
        
        for keyword in medical_keywords:
            features.append(1 if keyword in text_lower else 0)
        
        # Pad or truncate to 768 dimensions
        while len(features) < 768:
            features.append(0.0)
        
        return np.array(features[:768])
    
    def standardize_symptoms_with_similarity(self, symptom_text: str, threshold: float = 0.6) -> List[Dict]:
        """Standardize symptoms using similarity or pattern matching"""
        if pd.isna(symptom_text) or not symptom_text.strip():
            return []
        
        # Split multiple symptoms
        symptom_parts = re.split(r'[;,]|\s+or\s+|\s+and\s+', str(symptom_text).lower())
        
        standardized_symptoms = []
        
        for symptom in symptom_parts:
            symptom = symptom.strip()
            if len(symptom) < 3:
                continue
            
            # Try embedding-based similarity
            if self.sentence_model and self.symptom_embeddings is not None:
                try:
                    symptom_embedding = self.sentence_model.encode([symptom])
                    similarities = cosine_similarity(symptom_embedding, self.symptom_embeddings)[0]
                    
                    best_idx = np.argmax(similarities)
                    best_similarity = similarities[best_idx]
                    
                    if best_similarity >= threshold:
                        standardized_symptoms.append({
                            'original': symptom,
                            'standardized': self.symptom_labels[best_idx],
                            'confidence': float(best_similarity),
                            'embedding': self.symptom_embeddings[best_idx]
                        })
                        continue
                except Exception as e:
                    print(f"Embedding similarity error: {e}")
            
            # Fallback to pattern matching
            matched = self._pattern_based_symptom_matching(symptom)
            standardized_symptoms.append(matched)
        
        return standardized_symptoms
    
    def _pattern_based_symptom_matching(self, symptom: str) -> Dict:
        """Enhanced pattern-based symptom matching"""
        symptom_lower = symptom.lower().strip()
        
        # Define symptom patterns
        symptom_patterns = {
            'headache': ['head', 'headache', 'migraine', 'cephalgia'],
            'body ache': ['body', 'muscle', 'ache', 'myalgia', 'arthralgia'],
            'fever': ['fever', 'temperature', 'pyrexia', 'hyperthermia', 'febrile'],
            'nausea': ['nausea', 'nauseous', 'queasy', 'sick', 'vomit'],
            'cough': ['cough', 'coughing', 'tussis'],
            'abdominal pain': ['stomach', 'abdominal', 'belly', 'gastric', 'epigastric'],
            'chest pain': ['chest', 'thoracic', 'cardiac', 'heart'],
            'back pain': ['back', 'spine', 'spinal', 'lumbar'],
            'joint pain': ['joint', 'arthritis', 'rheumatic'],
            'postoperative pain': ['surgery', 'surgical', 'post-surgical', 'postoperative', 'post-op'],
            'dental pain': ['dental', 'tooth', 'teeth', 'oral', 'gum'],
            'burn pain': ['burn', 'scald', 'thermal'],
            'injury pain': ['injury', 'trauma', 'accident', 'wound'],
            'allergy': ['allergy', 'allergic', 'reaction', 'hives'],
            'cold symptoms': ['cold', 'runny nose', 'congestion', 'sneezing'],
            'anxiety': ['anxiety', 'anxious', 'panic', 'worry'],
            'insomnia': ['insomnia', 'sleep', 'sleepless', 'awake']
        }
        
        # Find best match
        best_match = 'general pain'
        best_confidence = 0.5
        
        for standard_symptom, keywords in symptom_patterns.items():
            for keyword in keywords:
                if keyword in symptom_lower:
                    # Calculate confidence based on exact match
                    if keyword == symptom_lower:
                        confidence = 0.9
                    elif symptom_lower.startswith(keyword) or symptom_lower.endswith(keyword):
                        confidence = 0.8
                    else:
                        confidence = 0.7
                    
                    if confidence > best_confidence:
                        best_match = standard_symptom
                        best_confidence = confidence
                        break
        
        return {
            'original': symptom,
            'standardized': best_match,
            'confidence': best_confidence,
            'embedding': np.zeros(384)
        }
    
    def extract_drug_entities(self, drug_text: str) -> List[Dict]:
        """Extract and process drug information"""
        if pd.isna(drug_text) or not drug_text.strip():
            return [{'name': 'unknown', 'embedding': np.zeros(768), 'original_text': ''}]
        
        entities = self.extract_medical_entities(str(drug_text))
        drugs = []
        
        # Get detected drug names
        drug_names = entities.get('DRUG', [])
        
        # If no entities detected, split by common delimiters
        if not drug_names:
            drug_names = [d.strip() for d in str(drug_text).split(',')]
            drug_names = [d for d in drug_names if d and len(d) > 1]
        
        # If still no drugs, use original text
        if not drug_names:
            drug_names = [str(drug_text).strip()]
        
        for drug in drug_names:
            if drug and len(drug) > 1:
                drug_embedding = self.get_clinical_embedding(drug)
                
                drugs.append({
                    'name': drug,
                    'embedding': drug_embedding,
                    'original_text': drug_text
                })
        
        return drugs if drugs else [{'name': 'unknown', 'embedding': np.zeros(768), 'original_text': drug_text}]
    
    def process_contraindications(self, contraindication_text: str) -> Dict:
        """Process contraindications with enhanced extraction"""
        if pd.isna(contraindication_text):
            return {
                'conditions': [],
                'warnings': [],
                'embedding': np.zeros(768),
                'original_text': ''
            }
        
        contraindication_str = str(contraindication_text)
        entities = self.extract_medical_entities(contraindication_str)
        
        # Extract conditions
        conditions = entities.get('CONDITION', [])
        
        # Get embedding for full text
        full_embedding = self.get_clinical_embedding(contraindication_str)
        
        # Split warnings by delimiters
        warnings = re.split(r'[;,]|\s+or\s+|\s+and\s+', contraindication_str.lower())
        warnings = [w.strip() for w in warnings if w.strip() and len(w) > 2]
        
        return {
            'conditions': conditions,
            'warnings': warnings,
            'embedding': full_embedding,
            'original_text': contraindication_str
        }
    
    def preprocess_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main preprocessing function with enhanced error handling"""
        print("Starting comprehensive medical NLP preprocessing...")
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Validate required columns
        required_cols = ['symptoms', 'Drug Name(s)', 'GENDER', 'Severity']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Warning: Missing required columns: {missing_cols}")
        
        processed_rows = []
        total_rows = len(df)
        
        for idx, row in df.iterrows():
            if idx % 10 == 0:
                print(f"Processing row {idx+1}/{total_rows} ({(idx+1)/total_rows*100:.1f}%)")
            
            try:
                # Extract and standardize symptoms
                symptoms = self.standardize_symptoms_with_similarity(row.get('symptoms', ''))
                
                if not symptoms:
                    symptoms = [{
                        'original': str(row.get('symptoms', 'unknown')),
                        'standardized': 'unknown',
                        'confidence': 0.0,
                        'embedding': np.zeros(384)
                    }]
                
                # Process drugs
                drugs = self.extract_drug_entities(row.get('Drug Name(s)', ''))
                
                # Process contraindications
                contraindications = self.process_contraindications(row.get('Contraindications / Cautions', ''))
                
                # Create rows for each symptom-drug combination
                for symptom in symptoms:
                    for drug in drugs:
                        new_row = {
                            # Original columns (with safe access)
                            'gender': row.get('GENDER', 'unknown'),
                            'symptom_category': row.get('Symptom Category', 'unknown'),
                            'original_symptoms': row.get('symptoms', ''),
                            'severity': row.get('Severity', 'unknown'),
                            'drug_class': row.get('Drug Class', 'unknown'),
                            'original_drug_names': row.get('Drug Name(s)', ''),
                            'route': row.get('Route', 'unknown'),
                            'otc_prescription': row.get('OTC/Prescription', 'unknown'),
                            'age_restrictions': row.get('Age Restrictions', ''),
                            'contraindications': row.get('Contraindications / Cautions', ''),
                            'follow_up_questions': row.get('Follow-up Question(s)', ''),
                            'recommendation': row.get('Recommendation', ''),
                            
                            # Processed columns
                            'standardized_symptom': symptom['standardized'],
                            'symptom_confidence': symptom['confidence'],
                            'symptom_original': symptom['original'],
                            'drug_name': drug['name'],
                            'contraindication_conditions': contraindications['conditions'],
                            'contraindication_warnings': contraindications['warnings'],
                            
                            # Embeddings
                            'symptom_embedding': symptom.get('embedding', np.zeros(384)).tolist(),
                            'drug_embedding': drug.get('embedding', np.zeros(768)).tolist(),
                            'contraindication_embedding': contraindications.get('embedding', np.zeros(768)).tolist(),
                            
                            # Additional features
                            'is_otc': str(row.get('OTC/Prescription', '')).lower() == 'otc',
                            'requires_prescription': str(row.get('OTC/Prescription', '')).lower() == 'prescription',
                            'requires_consultation': 'consult' in str(row.get('Recommendation', '')).lower(),
                        }
                        
                        processed_rows.append(new_row)
                        
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                continue
        
        processed_df = pd.DataFrame(processed_rows)
        
        # Remove duplicates
        if len(processed_df) > 0:
            duplicate_cols = ['gender', 'symptom_category', 'standardized_symptom', 'drug_name', 'severity']
            available_cols = [col for col in duplicate_cols if col in processed_df.columns]
            processed_df = processed_df.drop_duplicates(subset=available_cols)
        
        print(f"\n=== PREPROCESSING COMPLETED ===")
        print(f"Original rows: {total_rows}")
        print(f"Processed rows: {len(processed_df)}")
        if len(processed_df) > 0:
            print(f"Unique standardized symptoms: {processed_df['standardized_symptom'].nunique()}")
            print(f"Unique drugs: {processed_df['drug_name'].nunique()}")
            print(f"Average symptom confidence: {processed_df['symptom_confidence'].mean():.3f}")
        
        return processed_df
    
    def save_processed_data(self, df: pd.DataFrame, output_path: str = 'processed_medical_dataset.xlsx'):
        """Save processed data with comprehensive sheets"""
        if len(df) == 0:
            print("⚠ No data to save")
            return
        
        try:
            # Prepare data for Excel
            df_to_save = df.copy()
            
            # Convert embedding lists to strings for Excel
            embedding_cols = ['symptom_embedding', 'drug_embedding', 'contraindication_embedding']
            for col in embedding_cols:
                if col in df_to_save.columns:
                    df_to_save[col] = df_to_save[col].apply(
                        lambda x: ','.join(map(str, x)) if isinstance(x, list) else str(x)
                    )
            
            # Save to Excel with multiple sheets
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # Main dataset
                df_to_save.to_excel(writer, sheet_name='Processed_Dataset', index=False)
                
                # Summary statistics
                summary_data = {
                    'Total Records': len(df),
                    'Unique Symptoms': df['standardized_symptom'].nunique(),
                    'Unique Drugs': df['drug_name'].nunique(),
                    'Avg Symptom Confidence': df['symptom_confidence'].mean(),
                    'High Confidence (>0.8)': (df['symptom_confidence'] > 0.8).sum(),
                    'Low Confidence (<0.5)': (df['symptom_confidence'] < 0.5).sum(),
                    'OTC Medications': df['is_otc'].sum(),
                    'Prescription Medications': df['requires_prescription'].sum()
                }
                
                summary_df = pd.DataFrame(list(summary_data.items()), columns=['Metric', 'Value'])
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Symptom mapping
                symptom_mapping = df[['symptom_original', 'standardized_symptom', 'symptom_confidence']].drop_duplicates()
                symptom_mapping.to_excel(writer, sheet_name='Symptom_Mapping', index=False)
                
                # Drug information
                drug_info = df[['original_drug_names', 'drug_name', 'drug_class', 'route']].drop_duplicates()
                drug_info.to_excel(writer, sheet_name='Drug_Info', index=False)
                
                # Top symptoms and drugs
                top_symptoms = df['standardized_symptom'].value_counts().head(20)
                top_drugs = df['drug_name'].value_counts().head(20)
                
                pd.DataFrame({'Symptom': top_symptoms.index, 'Count': top_symptoms.values}).to_excel(
                    writer, sheet_name='Top_Symptoms', index=False)
                pd.DataFrame({'Drug': top_drugs.index, 'Count': top_drugs.values}).to_excel(
                    writer, sheet_name='Top_Drugs', index=False)
            
            print(f"✓ Processed dataset saved to: {output_path}")
            
            # Save embeddings separately for ML use
            try:
                embeddings_data = {
                    'symptom_embeddings': np.array(df['symptom_embedding'].tolist()),
                    'drug_embeddings': np.array(df['drug_embedding'].tolist()),
                    'contraindication_embeddings': np.array(df['contraindication_embedding'].tolist()),
                    'labels': df[['standardized_symptom', 'drug_name', 'severity']].values
                }
                
                embeddings_file = output_path.replace('.xlsx', '_embeddings.npz')
                np.savez(embeddings_file, **embeddings_data)
                print(f"✓ Embeddings saved separately: {embeddings_file}")
            except Exception as e:
                print(f"⚠ Could not save embeddings: {e}")
                
        except Exception as e:
            print(f"Error saving data: {e}")
            # Fallback: save as CSV
            csv_path = output_path.replace('.xlsx', '.csv')
            df.to_csv(csv_path, index=False)
            print(f"✓ Data saved as CSV fallback: {csv_path}")

def install_requirements():
    """Install required packages with better error handling"""
    import sys
    import subprocess
    
    # Check if we're in a virtual environment
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    
    required_packages = [
        'pandas',
        'numpy', 
        'scikit-learn',
        'openpyxl'
    ]
    
    optional_packages = [
        'transformers',
        'sentence-transformers', 
        'torch',
        'spacy'
    ]
    
    print("Installing required packages...")
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✓ {package} already installed")
        except ImportError:
            try:
                # Use --user only if NOT in virtual environment
                install_cmd = [sys.executable, "-m", "pip", "install", package]
                if not in_venv:
                    install_cmd.append("--user")
                
                subprocess.check_call(install_cmd)
                print(f"✓ {package} installed successfully")
            except subprocess.CalledProcessError:
                print(f"✗ Failed to install {package}")
                return False
    
    print("\nTrying to install optional packages (for advanced features)...")
    for package in optional_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✓ {package} already available")
        except ImportError:
            try:
                # Use --user only if NOT in virtual environment
                install_cmd = [sys.executable, "-m", "pip", "install", package]
                if not in_venv:
                    install_cmd.append("--user")
                
                subprocess.check_call(install_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f"✓ {package} installed")
            except subprocess.CalledProcessError:
                print(f"⚠ {package} not installed (optional)")
    
    return True

def main():
    """Main execution function with comprehensive error handling"""
    print("=== MEDICAL NLP PREPROCESSOR ===")
    
    # Install requirements
    if not install_requirements():
        print("Failed to install required packages. Please install manually:")
        print("pip install pandas numpy scikit-learn openpyxl")
        return None
    
    # Ask user about advanced models
    use_advanced = input("\nDo you want to use advanced medical models? (y/n, default=n): ").lower().strip()
    use_advanced_models = use_advanced in ['y', 'yes', '1', 'true']
    
    if use_advanced_models:
        print("Advanced models will be loaded (this may take time and require internet)")
    else:
        print("Using basic models only (faster, works offline)")
    
    # Initialize preprocessor
    try:
        preprocessor = MedicalNLPPreprocessor(use_advanced_models=use_advanced_models)
        print("✓ Preprocessor initialized successfully")
    except Exception as e:
        print(f"Error initializing preprocessor: {e}")
        return None
    
    # Load data
    data_file = input("\nEnter the path to your Excel file (or press Enter for default): ").strip()
    if not data_file:
        data_file = r'C:\Users\Supriya S\OneDrive\Desktop\IDP\Data from books_updated.xlsx'
    
    try:
        if not os.path.exists(data_file):
            print(f"File not found: {data_file}")
            return None
            
        df = pd.read_excel(data_file)
        print(f"✓ Dataset loaded: {len(df)} rows, {len(df.columns)} columns")
        print(f"Columns: {list(df.columns)}")
        
        # Show sample data
        print(f"\nFirst few rows:")
        print(df.head(3))
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
    # Preprocess dataset
    try:
        processed_df = preprocessor.preprocess_dataset(df)
        
        if len(processed_df) == 0:
            print("No data was processed successfully")
            return None
            
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return None
    
    # Display results
    print(f"\n=== PREPROCESSING RESULTS ===")
    print(f"Original dataset: {len(df)} rows")
    print(f"Processed dataset: {len(processed_df)} rows")
    print(f"Unique symptoms: {processed_df['standardized_symptom'].nunique()}")
    print(f"Unique drugs: {processed_df['drug_name'].nunique()}")
    print(f"Average symptom confidence: {processed_df['symptom_confidence'].mean():.3f}")
    
    # Show top symptoms and drugs
    print(f"\nTop 10 standardized symptoms:")
    print(processed_df['standardized_symptom'].value_counts().head(10))
    
    print(f"\nTop 10 drugs:")
    print(processed_df['drug_name'].value_counts().head(10))
    
    # Show sample processed data
    print(f"\nSample processed data:")
    sample_cols = [
        'gender', 'standardized_symptom', 'symptom_confidence', 
        'drug_name', 'severity', 'is_otc', 'requires_consultation'
    ]
    available_cols = [col for col in sample_cols if col in processed_df.columns]
    print(processed_df[available_cols].head(5))
    
    # Save processed data
    output_file = input(f"\nEnter output filename (default: processed_medical_dataset.xlsx): ").strip()
    if not output_file:
        output_file = 'processed_medical_dataset.xlsx'
    
    try:
        preprocessor.save_processed_data(processed_df, output_file)
        print(f"\n✓ Processing completed successfully!")
        print(f"Output saved to: {output_file}")
        
        # Show file location
        full_path = os.path.abspath(output_file)
        print(f"Full path: {full_path}")
        
    except Exception as e:
        print(f"Error saving results: {e}")
    
    return processed_df

if __name__ == "__main__":
    try:
        result = main()
        if result is not None:
            print(f"\n=== SUCCESS ===")
            print(f"Medical NLP preprocessing completed successfully!")
            print(f"Processed {len(result)} records")
        else:
            print(f"\n=== FAILED ===")
            print(f"Processing failed. Check error messages above.")
    except KeyboardInterrupt:
        print(f"\n\nProcessing interrupted by user.")
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()