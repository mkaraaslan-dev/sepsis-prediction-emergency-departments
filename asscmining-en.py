import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from mlxtend.frequent_patterns import apriori, association_rules
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# 1. SETTINGS & DATA LOADING
# ============================================================
FILE_POS = "sepsis pozitif grup_e.xlsx"
FILE_NEG = "sepsis negatif grup_e.xlsx"
MIN_SUPPORT = 0.25      # Nadir durumları ele
MIN_CONFIDENCE = 0.80   # Güvenilirlik sınırı
MIN_LIFT = 2.0          # İlişki gücü

print("--- STEP 1: DATA VALIDATION & ASSOCIATION MINING STARTED ---")
print("Reading files...")

try:
    df_pos = pd.read_excel(FILE_POS)
    df_neg = pd.read_excel(FILE_NEG)
    df = pd.concat([df_pos, df_neg], ignore_index=True)
except Exception as e:
    print(f"ERROR: Could not read file. {e}")
    exit()

# List of columns including GENDER (CİNSİYET)
target_columns_tr = [
    'YAŞ', 'CİNSİYET', 'NABIZ', 'SKB', 'DKB', 'OAB', 'SOLUNUM SAYISI', 
    'ATEŞ', 'SATURASYON', 'GKS', 'LAKTAT', 'PH', 'PCO2', 'BE', 
    'WBC', 'PLT', 'GLUKOZ', 'KRE', 'ÜRE', 'T.BİL', 'D.BİL'
]

# Dictionary to map Turkish headers to English academic abbreviations
col_mapping = {
    'YAŞ': 'Age',
    'CİNSİYET': 'Gender',  # Cinsiyet Eklendi
    'NABIZ': 'HR',         
    'SKB': 'SBP',          
    'DKB': 'DBP',          
    'OAB': 'MAP',          
    'SOLUNUM SAYISI': 'RR',
    'ATEŞ': 'Temp',        
    'SATURASYON': 'SpO2',
    'GKS': 'GCS',          
    'LAKTAT': 'Lactate',
    'PH': 'pH',
    'PCO2': 'PCO2',
    'BE': 'BE',            
    'WBC': 'WBC',
    'PLT': 'PLT',
    'GLUKOZ': 'Glucose',
    'KRE': 'Creatinine',
    'ÜRE': 'Urea',
    'T.BİL': 'T.Bil',
    'D.BİL': 'D.Bil'
}

# Select existing columns and rename them
cols_existing = [c for c in target_columns_tr if c in df.columns]
df_clean = df[cols_existing].rename(columns=col_mapping)

# ============================================================
# 2. SPECIAL HANDLING FOR GENDER & STRICT CLEANING
# ============================================================
print("Preprocessing Gender and cleaning numeric data...")

# 2.1. Handle Gender (Map Turkish/Numeric to English Categories)
if 'Gender' in df_clean.columns:
    # Standartlaştırma
    df_clean['Gender'] = df_clean['Gender'].astype(str).str.upper().str.strip()
    
    gender_map = {
        'E': 'Male', 'ERKEK': 'Male', 'M': 'Male', 'MALE': 'Male', 
        '1': 'Male', '1.0': 'Male',
        'K': 'Female', 'KADIN': 'Female', 'F': 'Female', 'FEMALE': 'Female', 
        '0': 'Female', '0.0': 'Female'
    }
    df_clean['Gender'] = df_clean['Gender'].map(gender_map)

# 2.2. Numeric Cleaning (Apply only to numeric columns, skip Gender)
numeric_cols = [c for c in df_clean.columns if c != 'Gender']

def strict_clean(val):
    if pd.isna(val): return np.nan
    val_str = str(val).strip().replace(',', '.')
    try:
        return float(val_str)
    except ValueError:
        return np.nan

for col in numeric_cols:
    df_clean[col] = df_clean[col].apply(strict_clean)

# PURE DATA: Drop rows with ANY missing value
df_pure = df_clean.dropna()
print(f"Total Data: {len(df)} -> Pure Data used for Mining: {len(df_pure)}")

if len(df_pure) < 10:
    print("WARNING: Too much data lost! Check your Gender column format or missing values.")
    exit()

# ============================================================
# 3. DISCRETIZATION (CATEGORIZATION)
# ============================================================
print("Discretizing variables...")
df_cat = pd.DataFrame()

def get_category_english(val, col):
    # Medical Thresholds
    if col == 'Lactate': return 'Lactate_HIGH' if val > 2 else 'Lactate_NORMAL'
    if col == 'pH': return 'pH_ACIDOSIS' if val < 7.35 else ('pH_ALKALOSIS' if val > 7.45 else 'pH_NORMAL')
    if col == 'PCO2': return 'PCO2_LOW' if val < 35 else ('PCO2_HIGH' if val > 45 else 'PCO2_NORMAL')
    if col == 'BE': return 'BE_LOW' if val < -2 else ('BE_HIGH' if val > 2 else 'BE_NORMAL')
    if col == 'Temp': return 'Temp_HIGH' if val > 38 else 'Temp_NORMAL'
    if col == 'SpO2': return 'SpO2_LOW' if val < 95 else 'SpO2_NORMAL'
    if col == 'Creatinine': return 'Creatinine_HIGH' if val > 1.2 else 'Creatinine_NORMAL'
    if col == 'Urea': return 'Urea_HIGH' if val > 50 else 'Urea_NORMAL'
    if col == 'WBC': return 'WBC_HIGH' if val > 12000 else ('WBC_LOW' if val < 4000 else 'WBC_NORMAL')
    if col == 'PLT': return 'PLT_LOW' if val < 150000 else 'PLT_NORMAL'
    if col == 'Glucose': return 'Glucose_HIGH' if val > 180 else ('Glucose_LOW' if val < 70 else 'Glucose_NORMAL')
    if col == 'GCS': return 'GCS_CRITICAL' if val < 9 else ('GCS_LOW' if val < 13 else 'GCS_NORMAL')
    if col == 'Age': return 'Age_ELDERLY' if val > 65 else 'Age_ADULT'
    return None

for col in df_pure.columns:
    # Special Handling for Gender (No Binning needed, just prefix)
    if col == 'Gender':
        df_cat[col] = 'Gender_' + df_pure[col].astype(str)
        continue

    # 1. Try Medical Rule
    temp = df_pure[col].apply(lambda x: get_category_english(x, col))
    
    # 2. If no rule, use Statistical Binning
    if temp.isnull().all():
        try:
            df_cat[col] = pd.qcut(df_pure[col], q=3, labels=[f'{col}_LOW', f'{col}_MEDIUM', f'{col}_HIGH'], duplicates='drop')
        except:
            df_cat[col] = pd.cut(df_pure[col], bins=3, labels=[f'{col}_LOW', f'{col}_MEDIUM', f'{col}_HIGH'])
    else:
        df_cat[col] = temp

# ============================================================
# 4. APRIORI ALGORITHM
# ============================================================
print("Calculating associations (Apriori)...")
df_encoded = pd.get_dummies(df_cat)
frequent_itemsets = apriori(df_encoded, min_support=MIN_SUPPORT, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=MIN_CONFIDENCE)

# Filter by Lift
rules_filtered = rules[rules['lift'] > MIN_LIFT].sort_values(by='lift', ascending=False)

# ============================================================
# 5. EXPORT TO EXCEL
# ============================================================
def clean_set(x):
    return ", ".join(list(x))

output_df = rules_filtered.copy()
output_df['antecedents'] = output_df['antecedents'].apply(clean_set)
output_df['consequents'] = output_df['consequents'].apply(clean_set)

# Rename columns
output_df = output_df.rename(columns={
    'antecedents': 'Antecedents (If)',
    'consequents': 'Consequents (Then)',
    'lift': 'Lift',
    'confidence': 'Confidence',
    'support': 'Support'
})

cols_export = ['Antecedents (If)', 'Consequents (Then)', 'Lift', 'Confidence', 'Support']

# STANDART DOSYA İSMİ
output_file = "Association_Rules_Analysis.xlsx" 
output_df[cols_export].to_excel(output_file, index=False)
print(f"Table saved: '{output_file}'")

# ============================================================
# 6. VISUALIZATION
# ============================================================
print("Plotting network graph...")
plt.figure(figsize=(18, 14))
G = nx.DiGraph()

# Plot Top 40 Rules
top_rules = rules_filtered.head(40)

for i, row in top_rules.iterrows():
    ants = list(row['antecedents'])
    cons = list(row['consequents'])
    weight = row['lift']
    
    ant_label = "\n+\n".join([x.replace('_', '\n') for x in ants])
    con_label = "\n".join([x.replace('_', '\n') for x in cons])
    
    G.add_edge(ant_label, con_label, weight=weight)

pos = nx.spring_layout(G, k=0.8, iterations=60, seed=42)
degrees = dict(G.degree)
node_sizes = [v * 200 + 1500 for v in degrees.values()] 

nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='#87CEFA', edgecolors='black', alpha=0.9)
nx.draw_networkx_edges(G, pos, edge_color='gray', arrowstyle='-|>', arrowsize=25, width=2, alpha=0.6)
nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold', 
                        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3', alpha=0.9))

plt.title(f"Physiological Consistency Analysis\nLIFT > {MIN_LIFT} | Confidence > {MIN_CONFIDENCE}", fontsize=20)
plt.axis('off')
plt.tight_layout()

# STANDART GRAFİK İSMİ
output_img = "Association_Network_Graph.png"
plt.savefig(output_img, dpi=300)
print(f"Graph saved: '{output_img}'")
print("--- PROCESS COMPLETED ---")