import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# 1. SETTINGS & SETUP
# ============================================================
print("--- PATIENT-BASED MISSING DATA ANALYSIS (ROW-WISE) ---")

FILE_POS = "sepsis pozitif grup_e.xlsx"
FILE_NEG = "sepsis negatif grup_e.xlsx"

# ADDED 'CİNSİYET_KODLU' to make it 21 features
NUMERIC_COLS_TR = [
    'YAŞ', 'NABIZ', 'SKB', 'DKB', 'OAB', 'SOLUNUM SAYISI', 
    'ATEŞ', 'SATURASYON', 'GKS', 'LAKTAT', 'PH', 'PCO2', 'BE', 
    'WBC', 'PLT', 'GLUKOZ', 'KRE', 'ÜRE', 'T.BİL', 'D.BİL',
    'CİNSİYET_KODLU' # Added Missing Column
]

# ============================================================
# 2. DATA LOADING & CLEANING
# ============================================================
try:
    df_pos = pd.read_excel(FILE_POS)
    df_neg = pd.read_excel(FILE_NEG)
    
    df_pos['TARGET'] = 1  # Sepsis
    df_neg['TARGET'] = 0  # Control
    
    df = pd.concat([df_pos, df_neg], ignore_index=True)
    
except Exception as e:
    print(f"ERROR: {e}")
    exit()

# Handle Gender Mapping if exists as String (E/K -> 0/1)
if 'CİNSİYET' in df.columns and 'CİNSİYET_KODLU' not in df.columns:
     df['CİNSİYET_KODLU'] = df['CİNSİYET'].astype(str).str.upper().map({'E': 0, 'K': 1, 'M': 0, 'F': 1})

# Numeric Conversion (Comma -> Dot)
cols_in_df = [c for c in NUMERIC_COLS_TR if c in df.columns]

for col in cols_in_df:
    # Force convert to numeric, errors become NaN
    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')

# ============================================================
# 3. ROW-WISE MISSING CHECK
# ============================================================
# Logic: If a row has even ONE NaN value in target columns, it is "Missing"
df['Has_Missing'] = df[cols_in_df].isnull().any(axis=1)

# Groups
pos_group = df[df['TARGET'] == 1]
neg_group = df[df['TARGET'] == 0]

# Calculate Statistics
stats = {
    "Group": ["Overall Total", "Sepsis (Positive)", "Control (Negative)"],
    
    "Total Patients": [
        len(df), 
        len(pos_group), 
        len(neg_group)
    ],
    
    "Complete Cases (Usable)": [
        len(df[~df['Has_Missing']]), # ~ means NOT
        len(pos_group[~pos_group['Has_Missing']]),
        len(neg_group[~neg_group['Has_Missing']])
    ],
    
    "Missing Cases (Incomplete)": [
        len(df[df['Has_Missing']]),
        len(pos_group[pos_group['Has_Missing']]),
        len(neg_group[neg_group['Has_Missing']])
    ]
}

df_stats = pd.DataFrame(stats)

# Calculate Percentage
df_stats["Missing Ratio (%)"] = (df_stats["Missing Cases (Incomplete)"] / df_stats["Total Patients"]) * 100

# ============================================================
# 4. REPORTING (ENGLISH)
# ============================================================
print("\n" + "="*60)
print("DATASET MISSINGNESS DISTRIBUTION REPORT")
print("Note: If a patient has even 1 missing feature, they are counted as 'Missing Cases'.")
print("="*60)
print(df_stats.to_string(index=False))
print("-" * 60)

# Export to Excel
df_stats.to_excel("Patient_Based_Missing_Analysis.xlsx", index=False)
print("Table saved: 'Patient_Based_Missing_Analysis.xlsx'")

# ============================================================
# 5. VISUALIZATION (Stacked Bar Chart - English)
# ============================================================
labels = ['Overall', 'Sepsis (+)', 'Control (-)']
complete_data = df_stats["Complete Cases (Usable)"].values
missing_data = df_stats["Missing Cases (Incomplete)"].values

x = np.arange(len(labels))
width = 0.5 

fig, ax = plt.subplots(figsize=(10, 7))

# Bars
bar1 = ax.bar(x, complete_data, width, label='Complete Data (Clean)', color='#2ca02c') # Green
bar2 = ax.bar(x, missing_data, width, bottom=complete_data, label='Missing Data (Needs Imputation)', color='#d62728') # Red

# Labels and Title
ax.set_ylabel('Number of Patients', fontsize=12)
ax.set_title('Distribution of Complete vs. Incomplete Patient Records', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.3)

# Function to add counts on bars
def add_labels(rects):
    for rect in rects:
        height = rect.get_height()
        if height > 0:
            ax.annotate(f'{int(height)}',
                        xy=(rect.get_x() + rect.get_width() / 2, rect.get_y() + height / 2),
                        xytext=(0, 0),
                        textcoords="offset points",
                        ha='center', va='center', color='white', fontweight='bold', fontsize=11)

add_labels(bar1)
add_labels(bar2)

plt.tight_layout()
output_img = "Patient_Based_Missing_Chart_English.png"
plt.savefig(output_img, dpi=300)
print(f"Chart saved: '{output_img}'")