import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import json
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# 1. HAM VERİ YÜKLEME (MGP'siz, Orijinal Dosyalar)
# ============================================================
print("--- STEP 3: FEATURE SELECTION (UPDATED / CLEAN DATA) ---")

FILE_POS = "sepsis pozitif grup_e.xlsx" 
FILE_NEG = "sepsis negatif grup_e.xlsx"

try:
    df_pos = pd.read_excel(FILE_POS)
    df_neg = pd.read_excel(FILE_NEG)
    
    # Etiketleme
    df_pos['TARGET'] = 1
    df_neg['TARGET'] = 0
    
    # Birleştirme
    df = pd.concat([df_pos, df_neg], ignore_index=True)
    print("Files merged successfully.")
    
except Exception as e:
    print(f"ERROR: Could not read files. Check filenames.\nDetails: {e}")
    exit()

# ============================================================
# 2. VERİ ÖN İŞLEME VE TİP DÖNÜŞÜMÜ
# ============================================================
# Türkçe Sütun İsimleri (Excel'deki hali)
target_numeric_columns = [
    'YAŞ', 'NABIZ', 'SKB', 'DKB', 'OAB', 'SOLUNUM SAYISI', 
    'ATEŞ', 'SATURASYON', 'GKS', 'LAKTAT', 'PH', 'PCO2', 'BE', 
    'WBC', 'PLT', 'GLUKOZ', 'KRE', 'ÜRE', 'T.BİL', 'D.BİL', 
    'CİNSİYET_KODLU'
]

# Cinsiyet kodlama (E/K -> 0/1)
if 'CİNSİYET' in df.columns:
    df['CİNSİYET_KODLU'] = df['CİNSİYET'].astype(str).str.upper().map({'E': 0, 'K': 1, 'M': 0, 'F': 1})

# Sadece hedef sütunları al
available_cols = [col for col in target_numeric_columns if col in df.columns]
X_raw = df[available_cols].copy()
y_raw = df['TARGET']

# --- STRING -> FLOAT DÖNÜŞÜMÜ ---
print("Converting data types to numeric...")
for col in X_raw.columns:
    # Virgülü nokta yap ve sayıya çevir
    X_raw[col] = pd.to_numeric(X_raw[col].astype(str).str.replace(',', '.'), errors='coerce')

# Target'ı geçici ekle (dropna için)
X_raw['TARGET'] = y_raw

# ============================================================
# 3. İSTATİSTİK RAPORLAMA (ENGLISH)
# ============================================================
total_rows = len(X_raw)
total_pos = len(X_raw[X_raw['TARGET'] == 1])
total_neg = len(X_raw[X_raw['TARGET'] == 0])

total_cells = X_raw[available_cols].size
missing_cells = X_raw[available_cols].isnull().sum().sum()
missing_ratio = (missing_cells / total_cells) * 100

print("\n" + "="*40)
print("DATASET GENERAL STATISTICS")
print("="*40)
print(f"Total Patients           : {total_rows}")
print(f" -> Positive (Sepsis)    : {total_pos}")
print(f" -> Negative (Control)   : {total_neg}")
print("-" * 40)
print(f"Total Data Cells         : {total_cells}")
print(f"Missing Cells (NaN)      : {missing_cells}")
print(f"Missing Ratio            : %{missing_ratio:.2f}")
print("="*40)

# ============================================================
# 4. TAM VERİ (LISTWISE DELETION) İLE FİLTRELEME
# ============================================================
# İçinde en az 1 tane NaN olan satırı komple sil
df_clean = X_raw.dropna()

print(f"\n'Listwise Deletion' applied for Feature Selection.")
print(f"Clean Rows Used for Analysis: {len(df_clean)} (Original: {total_rows})")

if len(df_clean) < 50:
    print("WARNING: Too much data lost! Results may not be reliable.")

X = df_clean[available_cols] # Target hariç özellikler
y = df_clean['TARGET']       # Sadece kalan satırların target'ı

# ============================================================
# 5. TÜRKÇE -> İNGİLİZCE ÇEVİRİ (AKADEMİK)
# ============================================================
# Bu sözlük sayesinde grafiklerde ve JSON dosyasında İngilizce isimler görünecek
rename_dict = {
    'YAŞ': 'Age',
    'NABIZ': 'Heart Rate',
    'SKB': 'SBP',
    'DKB': 'DBP',
    'OAB': 'MAP',
    'SOLUNUM SAYISI': 'Respiratory Rate',
    'ATEŞ': 'Temperature',
    'SATURASYON': 'SpO2',
    'GKS': 'GCS',
    'LAKTAT': 'Lactate',
    'PH': 'pH',
    'PCO2': 'PCO2',
    'BE': 'Base Excess',
    'WBC': 'WBC',
    'PLT': 'PLT',
    'GLUKOZ': 'Glucose',
    'KRE': 'Creatinine',
    'ÜRE': 'Urea',
    'T.BİL': 'Total Bilirubin',
    'D.BİL': 'Direct Bilirubin',
    'CİNSİYET_KODLU': 'Gender'
}

# Sütun isimlerini değiştir
X = X.rename(columns=rename_dict)
print("Column names translated to English.")

# ============================================================
# 6. RANDOM FOREST ILE SIRALAMA
# ============================================================
print("Calculating feature importance scores...")

rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X, y)

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

# Artık İngilizce sütun isimlerini kullanıyoruz
english_cols = X.columns
sorted_features = [english_cols[i] for i in indices]
sorted_importances = importances[indices]

# ============================================================
# 7. GÖRSELLEŞTİRME (ENGLISH)
# ============================================================
plt.figure(figsize=(12, 10))
sns.barplot(x=sorted_importances, y=sorted_features, palette="viridis")

plt.title(f'Most Important Features for Sepsis Prediction\n(Based on {len(df_clean)} Complete Cases)', fontsize=14)
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Features', fontsize=12)

plt.tight_layout()
output_img = 'Feature_Importance_Clean_English.png'
plt.savefig(output_img, dpi=300)
print(f"Chart saved: {output_img}")

# ============================================================
# 8. DENEY TASARIMI VE KAYIT
# ============================================================
# Kümeleri oluştur (İngilizce isimlerle)
feat_set_1 = sorted_features[:5]  # Top 5
feat_set_2 = sorted_features[:10] # Top 10
feat_set_3 = sorted_features      # Hepsi

print("\n" + "="*50)
print("FEATURE SETS (English)")
print("="*50)
print(f"SET_1 (Top 5)  : {feat_set_1}")
print(f"SET_2 (Top 10) : {feat_set_2}")
print(f"SET_3 (All)    : {len(feat_set_3)} features")
print("="*50)

# JSON Kaydet
sets_dictionary = {
    "SET_1": feat_set_1,
    "SET_2": feat_set_2,
    "SET_3": feat_set_3
}

with open("feature_sets.json", "w") as f:
    json.dump(sets_dictionary, f)

print("Lists saved to 'feature_sets.json' (English names).")