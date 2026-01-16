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
print("--- ADIM 3: GÜNCELLENMİŞ FEATURE SELECTION (SAF VERİ İLE) ---")

# BURAYA DİKKAT: Artık MGP dolu dosyaları değil, İLK HAM dosyaları kullanıyoruz
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
    print("Dosyalar başarıyla birleştirildi.")
    
except Exception as e:
    print(f"HATA: Dosyalar okunamadı. Dosya adlarını kontrol edin.\nHata detayı: {e}")
    exit()

# ============================================================
# 2. VERİ ÖN İŞLEME VE TİP DÖNÜŞÜMÜ
# ============================================================
# Analiz edilecek sayısal sütunlar
target_numeric_columns = [
    'YAŞ', 'NABIZ', 'SKB', 'DKB', 'OAB', 'SOLUNUM SAYISI', 
    'ATEŞ', 'SATURASYON', 'GKS', 'LAKTAT', 'PH', 'PCO2', 'BE', 
    'WBC', 'PLT', 'GLUKOZ', 'KRE', 'ÜRE', 'T.BİL', 'D.BİL', 
    'CİNSİYET_KODLU'
]

# Cinsiyet kodlama
if 'CİNSİYET' in df.columns:
    df['CİNSİYET_KODLU'] = df['CİNSİYET'].astype(str).str.upper().map({'E': 0, 'K': 1, 'M': 0, 'F': 1})
    # Haritalanamayanları NaN yapmamak için kontrol edebiliriz ama numeric conversion halleder

# Sadece hedef sütunları al
available_cols = [col for col in target_numeric_columns if col in df.columns]
X_raw = df[available_cols].copy()
y_raw = df['TARGET']

# --- STRING -> FLOAT DÖNÜŞÜMÜ ---
print("Veri tipleri sayısal formata çevriliyor...")
for col in X_raw.columns:
    # Virgülü nokta yap ve sayıya çevir. Hata veren (boşluk, harf vb.) NaN olur.
    X_raw[col] = pd.to_numeric(X_raw[col].astype(str).str.replace(',', '.'), errors='coerce')

# Target'ı X_raw içine geçici olarak ekleyelim ki dropna yaparken satır kaymasın
X_raw['TARGET'] = y_raw

# ============================================================
# 3. İSTATİSTİK RAPORLAMA (HOCALARIN İSTEDİĞİ KISIM)
# ============================================================
total_rows = len(X_raw)
total_pos = len(X_raw[X_raw['TARGET'] == 1])
total_neg = len(X_raw[X_raw['TARGET'] == 0])

# Toplam hücre sayısı ve eksik hücre sayısı
total_cells = X_raw[available_cols].size
missing_cells = X_raw[available_cols].isnull().sum().sum()
missing_ratio = (missing_cells / total_cells) * 100

print("\n" + "="*40)
print("VERİ SETİ GENEL İSTATİSTİKLERİ")
print("="*40)
print(f"Toplam Hasta Sayısı        : {total_rows}")
print(f" -> Pozitif (Sepsis)       : {total_pos}")
print(f" -> Negatif (Kontrol)      : {total_neg}")
print("-" * 40)
print(f"Toplam Veri Hücresi        : {total_cells}")
print(f"Toplam Eksik Hücre (NaN)   : {missing_cells}")
print(f"Eksik Veri Oranı           : %{missing_ratio:.2f}")
print("="*40)

# ============================================================
# 4. TAM VERİ (LISTWISE DELETION) İLE FİLTRELEME
# ============================================================
# İçinde en az 1 tane NaN olan satırı komple sil
df_clean = X_raw.dropna()

print(f"\nFeature Selection için 'Listwise Deletion' uygulandı.")
print(f"Analize Giren (Tam Dolu) Satır Sayısı: {len(df_clean)} (Orijinal: {total_rows})")

if len(df_clean) < 50:
    print("UYARI: Çok fazla veri silindi! Sonuçlar güvenilir olmayabilir.")
    print("Çözüm: 'target_numeric_columns' listesinden çok eksik olan sütunları çıkarın.")

X = df_clean[available_cols] # Target hariç özellikler
y = df_clean['TARGET']       # Sadece kalan satırların target'ı

# ============================================================
# 5. RANDOM FOREST ILE SIRALAMA (SAF VERİ ÜZERİNDE)
# ============================================================
print("Özellik önem skorları hesaplanıyor...")

rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X, y)

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

sorted_features = [available_cols[i] for i in indices]
sorted_importances = importances[indices]

# ============================================================
# 6. GÖRSELLEŞTİRME
# ============================================================
plt.figure(figsize=(12, 10))
sns.barplot(x=sorted_importances, y=sorted_features, palette="viridis")
plt.title(f'Sepsis En Önemli Parametreler\n(Sadece Tam Dolu {len(df_clean)} Hasta Verisiyle)', fontsize=14)
plt.xlabel('Önem Skoru')
plt.tight_layout()
plt.savefig('Adim3_Feature_Importance_Clean.png')
print("Grafik kaydedildi: Adim3_Feature_Importance_Clean.png")

# ============================================================
# 7. DENEY TASARIMI VE KAYIT
# ============================================================
# Kümeleri oluştur
feat_set_1 = sorted_features[:5]  # Top 5
feat_set_2 = sorted_features[:10] # Top 10
feat_set_3 = sorted_features      # Hepsi

print("\n" + "="*50)
print("GÜNCELLENMİŞ ÖZELLİK KÜMELERİ (Saf Veri Kaynaklı)")
print("="*50)
print(f"SET_1 (Top 5)  : {feat_set_1}")
print(f"SET_2 (Top 10) : {feat_set_2}")
print(f"SET_3 (Hepsi)  : {len(feat_set_3)} özellik")
print("="*50)

# JSON Kaydet
sets_dictionary = {
    "SET_1": feat_set_1,
    "SET_2": feat_set_2,
    "SET_3": feat_set_3
}

with open("feature_sets.json", "w") as f:
    json.dump(sets_dictionary, f)

print("Listeler 'feature_sets.json' dosyasına kaydedildi.")