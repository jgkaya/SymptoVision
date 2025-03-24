import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from skimage.feature import hog
import joblib

# Ultrason görüntülerinin bulunduğu klasörler
pcos_images_path = "proje/data/train/infected"
non_pcos_images_path = "proje/data/train/notinfected"

# Görselleri yükleme ve HOG özniteliklerini çıkarma
def load_images_and_features(folder, label):
    features = []
    labels = []
    
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Siyah beyaz olarak yükle
        img = cv2.resize(img, (128, 128))  # 128x128 boyutuna getir
        
        # HOG (Histogram of Oriented Gradients) ile özellik çıkar
        hog_features = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
        
        features.append(hog_features)
        labels.append(label)
    
    return features, labels

# PCOS ve Normal görüntüleri yükleyelim
pcos_features, pcos_labels = load_images_and_features(pcos_images_path, 1)
non_pcos_features, non_pcos_labels = load_images_and_features(non_pcos_images_path, 0)

# Veriyi birleştirme
X = np.array(pcos_features + non_pcos_features)
y = np.array(pcos_labels + non_pcos_labels)

# Eğitim ve test verisine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Veriyi ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# RandomForest Modelini Eğitme
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Modeli Kaydetme
joblib.dump(rf_model, "proje/pcos_ultrasound_ml_model.pkl")
joblib.dump(scaler, "proje/pcos_ultrasound_scaler.pkl")

# Modelin Doğruluğunu Yazdır
accuracy = rf_model.score(X_test_scaled, y_test)
print(f"Model Doğruluk: {accuracy:.2f}")
