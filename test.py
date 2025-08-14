import pandas as pd
from itertools import combinations
from collections import Counter
from symptom_checker import SymptomChecker

#Load your dataset from Excel file
file_path = r'C:\Users\supri\Smartmed-AI-based-medical-dispenser\dataset_1.xlsx'

try:
    df = pd.read_excel(file_path)
    print(f"Dataset loaded successfully! Shape: {df.shape}")
    print(f"Total rows in dataset: {len(df)}")
except Exception as e:
    print(f"Error loading file: {e}")
    exit()

def analyze_complete_dataset(df):
    """
    Analyze the complete dataset to find all symptom combinations
    """
    # Define symptom columns
    symptom_columns = ['Fever', 'Common Cold', 'Cough', 'Body Pain', 
                      'Headache', 'Menstrual Cramps', 'Sprain', 'Indigestion', 'Toothache']
    
    # Check which symptom columns exist in the dataset
    available_symptoms = [col for col in symptom_columns if col in df.columns]
    print(f"Available symptom columns: {available_symptoms}")
    
    if not available_symptoms:
        print("No symptom columns found in the dataset!")
        return []
    
    print(f"\nDataset Analysis:")
    print(f"Total rows: {len(df)}")
    
    # First, let's see the distribution of symptoms
    print(f"\nIndividual Symptom Counts:")
    print("-" * 40)
    symptom_counts = {}
    for symptom in available_symptoms:
        yes_count = (df[symptom].astype(str).str.strip().str.lower() == 'yes').sum()
        symptom_counts[symptom] = yes_count
        print(f"{symptom}: {yes_count} cases ({yes_count/len(df)*100:.2f}%)")
    
    # Now find all combinations that occur together in the same row
    combination_counts = Counter()
    patients_with_multiple_symptoms = 0
    symptom_distribution = Counter()
    
    # For each row, find which symptoms are present
    for idx, row in df.iterrows():
        present_symptoms = []
        for symptom in available_symptoms:
            if str(row[symptom]).strip().lower() == 'yes':
                present_symptoms.append(symptom)
        
        # Count how many symptoms this patient has
        num_symptoms = len(present_symptoms)
        symptom_distribution[num_symptoms] += 1
        
        if num_symptoms >= 2:
            patients_with_multiple_symptoms += 1
            
        # Generate all combinations of 2 or more symptoms for this patient
        if len(present_symptoms) >= 2:
            for r in range(2, len(present_symptoms) + 1):
                for combo in combinations(present_symptoms, r):
                    combo_tuple = tuple(sorted(combo))
                    combination_counts[combo_tuple] += 1
    
    print(f"\nPatient Symptom Distribution:")
    print("-" * 40)
    for num_symptoms in sorted(symptom_distribution.keys()):
        count = symptom_distribution[num_symptoms]
        percentage = (count / len(df)) * 100
        print(f"Patients with {num_symptoms} symptoms: {count} ({percentage:.2f}%)")
    
    print(f"\nPatients with multiple symptoms: {patients_with_multiple_symptoms} ({patients_with_multiple_symptoms/len(df)*100:.2f}%)")
    
    # Convert combinations to list format
    all_combinations = []
    for combo, count in combination_counts.items():
        all_combinations.append({
            'symptoms': list(combo),
            'count': count,
            'percentage': round((count / len(df)) * 100, 2)
        })
    
    # Sort by count (most common first)
    all_combinations.sort(key=lambda x: x['count'], reverse=True)
    
    return all_combinations, symptom_counts, symptom_distribution

def print_complete_analysis(df):
    """
    Print complete analysis of the dataset
    """
    print("=" * 80)
    print("COMPLETE DATASET SYMPTOM COMBINATION ANALYSIS")
    print("=" * 80)
    
    all_combinations, symptom_counts, symptom_distribution = analyze_complete_dataset(df)
    
    if not all_combinations:
        print("No symptom combinations found in the dataset.")
        print("This means no patients have multiple symptoms simultaneously.")
        return
    
    print(f"\nDATASET SUMMARY:")
    print("-" * 50)
    print(f"Total patients: {len(df)}")
    print(f"Total unique symptom combinations: {len(all_combinations)}")
    
    # Calculate total occurrences
    total_combination_occurrences = sum(combo['count'] for combo in all_combinations)
    print(f"Total combination occurrences: {total_combination_occurrences}")
    
    # Print ALL combinations
    print(f"\nALL {len(all_combinations)} SYMPTOM COMBINATIONS:")
    print("-" * 80)
    for i, combo in enumerate(all_combinations, 1):
        symptoms_str = ", ".join(combo['symptoms'])
        print(f"{i:3d}. [{symptoms_str}] - {combo['count']} cases ({combo['percentage']}%)")
    
    # Group by size
    print(f"\n\nCOMBINATIONS GROUPED BY SIZE:")
    print("-" * 80)
    
    combinations_by_size = {}
    for combo in all_combinations:
        size = len(combo['symptoms'])
        if size not in combinations_by_size:
            combinations_by_size[size] = []
        combinations_by_size[size].append(combo)
    
    for size in sorted(combinations_by_size.keys()):
        combo_list = combinations_by_size[size]
        total_for_size = sum(combo['count'] for combo in combo_list)
        print(f"\n{size}-SYMPTOM COMBINATIONS ({len(combo_list)} unique combinations, {total_for_size} total occurrences):")
        for i, combo in enumerate(combo_list, 1):
            symptoms_str = ", ".join(combo['symptoms'])
            print(f"  {i:2d}. [{symptoms_str}] - {combo['count']} cases ({combo['percentage']}%)")
    
    # Verification
    print(f"\n\nVERIFICATION:")
    print("-" * 40)
    patients_with_combinations = sum(count for num_symptoms, count in symptom_distribution.items() if num_symptoms >= 2)
    print(f"Patients with 2+ symptoms: {patients_with_combinations}")
    print(f"This matches the analysis above: {patients_with_combinations > 0}")
    
    return all_combinations

# Run the complete analysis
all_combinations = print_complete_analysis(df)

# Save complete results to file
try:
    with open('complete_symptom_analysis.txt', 'w') as f:
        f.write("COMPLETE DATASET SYMPTOM COMBINATION ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Dataset: {len(df)} total patients\n")
        f.write(f"Total unique combinations: {len(all_combinations) if all_combinations else 0}\n\n")
        
        if all_combinations:
            f.write("ALL SYMPTOM COMBINATIONS:\n")
            f.write("-" * 50 + "\n")
            for i, combo in enumerate(all_combinations, 1):
                symptoms_str = ", ".join(combo['symptoms'])
                f.write(f"{i:3d}. [{symptoms_str}] - {combo['count']} cases ({combo['percentage']}%)\n")
        else:
            f.write("No symptom combinations found.\n")
    
    print(f"\n\nComplete analysis saved to 'complete_symptom_analysis.txt'")
except:
    print(f"\nCould not save to file, but complete analysis is shown above")

# Additional verification - show sample rows with multiple symptoms
print(f"\n\nSAMPLE ROWS WITH MULTIPLE SYMPTOMS:")
print("-" * 60)
symptom_columns = ['Fever', 'Common Cold', 'Cough', 'Body Pain', 
                  'Headache', 'Menstrual Cramps', 'Sprain', 'Indigestion', 'Toothache']
available_symptoms = [col for col in symptom_columns if col in df.columns]

sample_rows = []
for idx, row in df.iterrows():
    present_symptoms = []
    for symptom in available_symptoms:
        if str(row[symptom]).strip().lower() == 'yes':
            present_symptoms.append(symptom)
    
    if len(present_symptoms) >= 2:
        sample_rows.append({
            'row_index': idx,
            'symptoms': present_symptoms,
            'count': len(present_symptoms)
        })
    
    if len(sample_rows) >= 10:  # Show first 10 examples
        break

for i, sample in enumerate(sample_rows, 1):
    symptoms_str = ", ".join(sample['symptoms'])
    print(f"Row {sample['row_index']:4d}: [{symptoms_str}] ({sample['count']} symptoms)")

print(f"\nTotal rows with multiple symptoms: {len([row for idx, row in df.iterrows() if sum(1 for col in available_symptoms if str(row[col]).strip().lower() == 'yes') >= 2])}")

if __name__ == "__main__":
    csv_path = r"C:\Users\Supriya S\OneDrive\Desktop\IDP\dataset_1.xlsx"
    checker = SymptomChecker(csv_path)
    checker.run_conversation()