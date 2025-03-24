import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from deap import base, creator, tools, algorithms
import random
import joblib
# PCOS veri setini yükleyelim ve ön işleme yapalım
pcos_data = pd.read_csv("proje/PCOS_data.csv")
label_pcos = pcos_data["PCOS (Y/N)"]

pcos_data.drop(["Sl. No", "Patient File No.", "PCOS (Y/N)", "Unnamed: 44", "II    beta-HCG(mIU/mL)", "Marraige Status (Yrs)","Fast food (Y/N)",
"Blood Group", "Pulse rate(bpm) ", "RR (breaths/min)", "Height(Cm) ", "Weight (Kg)", "BMI", "BP _Systolic (mmHg)", "BP _Diastolic (mmHg)","Reg.Exercise(Y/N)","Follicle No. (L)","Follicle No. (R)","Avg. F size (L) (mm)","Avg. F size (R) (mm)"], axis=1, inplace=True)

# "AMH(ng/mL)" sütununu sayısal bir forma çevirmeye çalışalım
pcos_data["AMH(ng/mL)"] = pd.to_numeric(pcos_data["AMH(ng/mL)"], errors='coerce')

# Eksik (NaN) değerleri ortalama ile dolduralım
pcos_data["AMH(ng/mL)"] = pcos_data["AMH(ng/mL)"].fillna(pcos_data["AMH(ng/mL)"].mean())

print(pcos_data.columns)
print(pcos_data.head())
# Eğitim ve test verisi olarak ayıralım
X_train, X_test, y_train, y_test = train_test_split(pcos_data, label_pcos, test_size=0.2, random_state=42)

# Verilerin ölçeklendirilmesi
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Genetik Algoritma - DEAP ayarları
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Bireyler (genler), her bir özellik için seçilip seçilmeyeceğini (0 veya 1) belirleyen bir liste olacak
n_features = X_train_scaled.shape[1]
toolbox.register("attr_bool", random.randint, 0, 1)

# Popülasyonun tanımlanması (bireylerin tanımlanması)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=n_features)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# RandomForestClassifier modeli ile doğruluğa dayalı fitness fonksiyonu
def eval_individual(individual):
    # Seçilen özelliklerin alt kümesini kullanarak modelin eğitilmesi
    selected_features = [index for index in range(len(individual)) if individual[index] == 1]
    if len(selected_features) == 0:
        return 0,
    
    X_train_subset = X_train_scaled[:, selected_features]
    X_test_subset = X_test_scaled[:, selected_features]
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_subset, y_train)
    predictions = model.predict(X_test_subset)
    
    accuracy = accuracy_score(y_test, predictions)
    return accuracy,

# Genetik Algoritma operatörlerinin tanımlanması
toolbox.register("evaluate", eval_individual)
toolbox.register("mate", tools.cxTwoPoint)  # Çiftleştirme (çaprazlama) yöntemi
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)  # Mutasyon
toolbox.register("select", tools.selTournament, tournsize=3)  # Seçim yöntemi

# Popülasyonun başlatılması ve genetik algoritmanın çalıştırılması
pop = toolbox.population(n=50)  # 50 bireyden oluşan bir popülasyon

# Genetik algoritmayı çalıştırma
algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=20, verbose=True)

# Sonuçların çıkarılması
best_individual = tools.selBest(pop, k=1)[0]
selected_features = [index for index in range(len(best_individual)) if best_individual[index] == 1]

print("En iyi özellik kombinasyonu:", selected_features)
selected_columns = pcos_data.columns[selected_features]
print("Seçilen özellikler:", selected_columns)



# Seçilen özelliklere göre veriyi filtreleme
X_train_selected = X_train_scaled[:, selected_features]  # Sadece seçilen özellikler
X_test_selected = X_test_scaled[:, selected_features]

# Veriyi ölçeklendirin (YENİ SCALER)
scaler = StandardScaler()
X_train_selected_scaled = scaler.fit_transform(X_train_selected)
X_test_selected_scaled = scaler.transform(X_test_selected)

# Modeli yeniden eğitme
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_selected_scaled, y_train)

# Modeli ve scaler'ı kaydetme
joblib.dump(rf_model, "proje/pcos_diagnosis_model.pkl")
joblib.dump(scaler, "proje/scaler.pkl")

# Modeli daha sonra yüklemek için
# rf_model = joblib.load("pcos_diagnosis_model.pkl")
