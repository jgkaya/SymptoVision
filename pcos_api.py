import gradio as gr
import joblib
import numpy as np
import cv2
import re
from skimage.feature import hog
from skimage.measure import label, regionprops
from openai import OpenAI
from PIL import Image

# **Eğitilmiş modelleri yükleme**
text_model = joblib.load("proje/pcos_diagnosis_model.pkl")
text_scaler = joblib.load("proje/scaler.pkl")
image_model = joblib.load("proje/pcos_ultrasound_ml_model.pkl")  # CNN modeli
image_scaler = joblib.load("proje/pcos_ultrasound_scaler.pkl")

# **GPT destekli PCOS teşhisi (Doğal Dil)**
def ask_openai_agent_with_comment(api_key, question, diagnosis=None):
    client = OpenAI(api_key=api_key)
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Sen bir doktor asistanısın. Hastaya açıklayıcı ama tıbbi bir yorum yap. Eğer PCOS teşhis edildiyse semptomları azaltmak için öneriler ver, edilmediyse genel sağlık önerileri sun."},
            {"role": "user", "content": f"{question}"},
            {"role": "assistant", "content": diagnosis}
        ],
        max_tokens=350
    )
    return response.choices[0].message.content

def predict_pcos_from_text(api_key, natural_language_input):
    diagnosis = "unknown" 
    question = (f"Bu açıklamaya göre gerekli parametreleri çıkar ve sayısal hale getir: {natural_language_input}.\n"
                "Çıktı sadece sayılar ve birim olmadan şu sırada olmalı: "
                "Yaş, döngü uzunluğu, hamile mi (0: Hayır, 1: Evet), düşük sayısı, "
                "LH seviyesi, kalça ölçüsü, bel:kalça oranı, kilo alma, "
                "kıl büyümesi, cilt koyulaşması, saç kaybı, endometrium (mm).")
    
    gpt_response = ask_openai_agent_with_comment(api_key, question, diagnosis)
    
    values = re.findall(r"[-+]?\d*\.\d+|\d+", gpt_response)  

    if len(values) != 12:
        return f"Yanıt düzgün formatta değil. Beklenen 12 değer, ancak {len(values)} değer bulundu: {values}", None

    input_data = np.array([float(value) for value in values]).reshape(1, -1)
    scaled_data = text_scaler.transform(input_data)
    prediction = text_model.predict(scaled_data)

    diagnosis = "Polikistik Over Sendromu Teşhis Edildi, Doktora Görünün" if prediction[0] == 1 else "Polikistik Over Sendromu Teşhis Edilmemiştir"

    # GPT'den teşhise özel yorum al
    comment = ask_openai_agent_with_comment(api_key, natural_language_input, diagnosis)

    return diagnosis, comment


# **Ultrason görüntüsünden folikül sayısı tespiti**
def detect_follicles(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)  # Görüntüyü yumuşatmak için Gaussian blur
    _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)  # İkili görüntüleme
    
    # Folikülleri tespit et
    labeled_image = label(thresh)
    regions = regionprops(labeled_image)
    
    follicle_count = sum(1 for region in regions if region.area >= 50)  # 50 pikselden büyük alanları folikül kabul et
    return follicle_count

def extract_features_from_image(image):
    # Görüntüyü gri tona çeviriyoruz
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    resized_image = cv2.resize(gray_image, (128, 128))

    # HOG özelliklerini çıkarıyoruz (multichannel argümanını kaldırıyoruz)
    features, hog_image = hog(resized_image, pixels_per_cell=(8, 8),
                              cells_per_block=(2, 2), visualize=True)

    return features  # HOG ile çıkarılan özellikleri döndürüyoruz

# **Görüntüden PCOS teşhisi (Ultrason Görseli)**
def predict_pcos_from_image(api_key, image_input):
    # Gradio'dan gelen image_input muhtemelen bir PIL.Image ya da numpy array olabilir.
    
    # Eğer resim bir numpy array ise
    if isinstance(image_input, np.ndarray):
        image = Image.fromarray(image_input)
    else:
        image = image_input  # Eğer zaten bir PIL.Image ise
    
    # Resmi işle
    resized_image = image.resize((128, 128))  # Örnek olarak boyutlandırma
    features = extract_features_from_image(np.array(resized_image))  # Özellik çıkarımı için array'e çevir
    
    # Özellikleri kullanarak tahmin yapın
    scaled_features = image_scaler.transform([features])  # Özellikleri scale et
    prediction = image_model.predict(scaled_features)  # PCOS teşhisi modeli

    # Eğer PCOS teşhisi edilmemişse folikül sayısını 0 olarak ayarla
    if prediction[0] == 0:
        follicle_count = 0
    else:
        follicle_count = detect_follicles(np.array(image))  # Folikül sayısını hesapla

    # Tahmin sonucunu döndür
    return "Polikistik Over Sendromu Teşhis Edildi, Doktora Görünün" if prediction[0] == 1 else "Polikistik Over Sendromu Teşhis Edilmemiştir", follicle_count

# **PCOS teşhisi - Metin ve Görsel sonuçlarını birleştir**
def predict_pcos(api_key, text_input=None, image_input=None):
    text_result = None
    image_result = None
    follicle_count = None

    gpt_comment = ""  # GPT'den alınan genel yorum
    
    # Metin bazlı teşhis
    if text_input:
        text_result, text_comment = predict_pcos_from_text(api_key, text_input)
        gpt_comment += f"Metin tabanlı analiz: {text_comment}\n\n"

    # Görüntü bazlı teşhis
    if image_input is not None:
        try:
            image_result, follicle_count = predict_pcos_from_image(api_key, image_input)
        except ValueError as e:
            return f"Hata: {str(e)}", None  # Hata durumunda iki değer döndür
    
        gpt_comment += f"Ultrason görüntü tabanlı analiz: Folikül sayısı {follicle_count} olarak tespit edildi.\n"
    
    # Sonuçları birleştir
    if text_result == "Polikistik Over Sendromu Teşhis Edildi, Doktora Görünün" or image_result == "Polikistik Over Sendromu Teşhis Edildi, Doktora Görünün":
        final_result = "Polikistik Over Sendromu Teşhis Edildi, Doktora Görünün"
    else:
        final_result = "Polikistik Over Sendromu Teşhis Edilmemiştir"

    follicle_message = f"Folikül Sayısı: {follicle_count}" if follicle_count is not None else ""

    # Son olarak GPT'den genel bir yorum alarak, teşhisi daha da detaylandır
    final_comment = ask_openai_agent_with_comment(api_key, f"PCOS teşhisi sonuçlarına göre genel bir açıklama yapar mısın?", final_result)

    return f"{final_result}\n\n{final_comment}", follicle_message


# Gradio Arayüzü - Göz alıcı ve modern tasarım
demo = gr.Interface(
    fn=predict_pcos,
    inputs=[
        gr.Textbox(label="OpenAI API Anahtarınızı girin", placeholder="API Anahtarı", type="password"),
        gr.Textbox(label="Belirtilerinizi açıklayın", placeholder="Yaşınızı, döngü uzunluğunuzu ve diğer belirtilerinizi girin", lines=4),
        gr.Image(type="numpy", label="Ultrason Görseli Yükleyin")
    ],
    outputs=[
        gr.Textbox(label="Teşhis Sonucu"),
        gr.Textbox(label="Folikül Sayısı")
    ],
    title="🩺 👩‍⚕️ PCOS Teşhisine Yardımcı Araç 👩‍⚕️ 🩺",
    description=(
        "🔍 Polikistik Over Sendromu (PCOS) teşhisine yardımcı olan bir araç. Belirtilerinizi yazın ve ultrason görüntüsünü yükleyin, "
        "metin ve görüntü tabanlı analiz sonuçlarıyla birlikte GPT destekli yorum alın."
    ),
    theme="soft",
    css="""
       /* Genel Yazı Rengi ve Font Ayarları */
body {
    color: #333 !important; /* Daha koyu bir metin rengi */
    font-size: 1.1rem !important; /* Font boyutu büyütüldü */
    margin: 0; /* Sayfa genelinde margin'i sıfırladık */
    padding: 0;
}

/* Başlıklar */
h1, h2, h3 {
    color: #1e1e1e !important; /* Başlıkları daha koyu yapıyoruz */
    font-weight: bold !important; /* Kalın başlıklar */
    font-size: 2.5rem !important; /* Başlık boyutunu büyük yaptık */
    margin-top: 20px !important; /* Başlıkların üst boşlukları */
    margin-bottom: 10px !important; /* Başlıkların alt boşlukları */
    text-align: center !important; /* Başlığı ortaya hizalıyoruz */
    background-color:rgb(84, 110, 242) !important; /* Başlık arka planını dikkat çekici yapıyoruz */
    color: white !important; /* Başlık yazı rengini beyaz yapıyoruz */
    padding: 20px !important; /* Başlığa ekstra boşluk ekliyoruz */
    border-radius: 10px !important; /* Başlık kenarlarını yuvarlatıyoruz */
}


/* Textarea ve Input Alanları */
textarea, input {
    color: #000 !important; /* Metin rengini siyah yapıyoruz */
    background-color: #fff !important; /* Arka plan beyaz */
    font-size: 1.1rem !important; /* Yazı boyutunu artırdık */
    padding: 10px !important; /* İç kenarlık (padding) */
    border: 1px solid #ccc !important; /* Kenarlık ekledik */
    outline: none !important; /* Focus olduğunda varsayılan mavi kenarlık kalkacak */
    border-radius: 5px !important; /* Kenarları yuvarlatıyoruz */
}

/* Placeholder Renkleri */
textarea::placeholder, input::placeholder {
    color: #888 !important; /* Placeholder rengini daha açık yapıyoruz */
}

/* Focus durumunda Input ve Textarea */
textarea:focus, input:focus {
    border-color: #f25c54 !important; /* Focus olduğunda kenarlık kırmızı */
    box-shadow: 0 0 5px rgba(242, 92, 84, 0.5) !important; /* Focus olduğunda hafif gölge */
}

/* Butonlar */
button {
    font-size: 1.2rem !important; /* Buton yazılarını büyütüyoruz */
    padding: 10px 20px !important; /* Buton iç boşlukları */
    border-radius: 5px !important; /* Kenarları yuvarlatıyoruz */
    color: white !important; /* Buton yazılarını beyaz yapıyoruz */
}

/* Submit Butonu */
button.submit {
    background-color: #f25c54 !important; /* Dikkat çekici bir renk */
}

button.submit:hover {
    background-color: #e63946 !important; /* Hover sırasında daha koyu renk */
}

/* Clear Butonu */
button.clear {
    background-color: #6c757d !important; /* Nötr bir renk */
    color: white !important;
}

button.clear:hover {
    background-color: #5a6268 !important; /* Hover sırasında daha koyu */
}

/* Görsel Yükleme Alanı (Dropzone) */
.dropzone {
    border: 2px dashed #4a4e69 !important; /* Sınır rengini koyu yapıyoruz */
    background-color: #f8f8ff !important; /* Arka planı açık renk tutuyoruz */
    color: #333 !important; /* Yazı rengini koyulaştırıyoruz */
    padding: 20px !important; /* İç boşluk artırıyoruz */
    border-radius: 10px !important; /* Kenarları yuvarlatıyoruz */
}
    """
)

# Uygulamayı başlat
demo.launch(share=True)
