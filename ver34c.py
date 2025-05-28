# -*- coding: utf-8 -*-
"""
Created on Tue May 27 14:38:22 2025

@author: Anis2
"""

# -*- coding: utf-8 -*-


# -*- coding: utf-8 -*-


import streamlit as st
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from datetime import datetime

# NLTK yüklemeleri
nltk.download('punkt')
nltk.download('stopwords')

# Veri Yükleme




column_names = [
    "sınıf", "ders_adı", "hoca-adı", "gün1", "saat1", "derslik1",
    "gün2", "saat2", "derslik2", "vizetarihi", "saat.1",
    "finaltarihi", "saat.2", "butunlemetarihi", "saat.3"
]
df = pd.read_excel('ders_bilgi.xlsx', names=column_names, header=1)







# Ön işleme
stop_words = set(stopwords.words('turkish'))

def preprocess_text(text):
    words = word_tokenize(str(text).lower())
    filtered = [w for w in words if w.isalnum() and w not in stop_words]
    return " ".join(filtered)

df['document'] = df.apply(lambda row: f"{row['sınıf']} {row['ders_adı']} {row['hoca-adı']} "
                                      f"{row['gün1']} {row['saat1']} {row['derslik1']} "
                                      f"{row['gün2']} {row['saat2']} {row['derslik2']} "
                                      f"{row['vizetarihi']} {row['saat.1']} "
                                      f"{row['finaltarihi']} {row['saat.2']} "
                                      f"{row['butunlemetarihi']} {row['saat.3']}", axis=1)

df['processed'] = df['document'].apply(preprocess_text)

# Vektörleştirme
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['processed'])

# Eşleşme bulucu
def find_best_match(user_input):
    processed_input = preprocess_text(user_input)
    user_vec = vectorizer.transform([processed_input])
    similarities = cosine_similarity(user_vec, tfidf_matrix)
    best_idx = similarities.argmax()
    return df.iloc[best_idx]

# Bilgi çıkarımı
def extract_info(question, row):
    q = question.lower()

    # Selamlaşmalar
    if re.search(r'\b(selam|merhaba|günaydın|iyi akşamlar|nasılsın)\b', q):
        return "Merhaba! Size nasıl yardımcı olabilirim? 😊"

    # Sınav Tarihleri
    if 'final' in q:
        return f"{row['ders_adı']} dersi finali: {row['finaltarihi']} Saat: {row['saat.2']}"
    if 'vize' in q:
        return f"{row['ders_adı']} dersi vizesi: {row['vizetarihi']} Saat: {row['saat.1']}"
    if 'büt' in q or 'bütünleme' in q:
        return f"{row['ders_adı']} dersi bütünlemesi: {row['butunlemetarihi']} Saat: {row['saat.3']}"

    # Sınıf bilgisi
    sinif_match = re.search(r'(\d)\.?\s*sınıf', q)
    if 'hangi sınıf' in q or 'kaçıncı sınıf' in q:
        return f"{row['ders_adı']} dersi {row['sınıf']}. sınıf dersidir."
    elif sinif_match:
        sinif = int(sinif_match.group(1))
        dersler = df[df['sınıf'] == sinif]['ders_adı'].unique()
        return f"{sinif}. sınıf dersleri:\n" + "\n".join(f"- {d}" for d in dersler)

    # Gün filtresi
    gunler = ['pazartesi', 'salı', 'çarşamba', 'perşembe', 'cuma']
    for gun in gunler:
        if gun in q:
            if sinif_match:
                sinif = int(sinif_match.group(1))
                dersler = df[((df['gün1'].str.lower() == gun) | (df['gün2'].str.lower() == gun)) & (df['sınıf'] == sinif)]
            else:
                dersler = df[(df['gün1'].str.lower() == gun) | (df['gün2'].str.lower() == gun)]
            if not dersler.empty:
                return f"{gun.capitalize()} günü dersler:\n" + "\n".join(f"- {d}" for d in dersler['ders_adı'].unique())
            return f"{gun.capitalize()} günü için ders bulunamadı."

    # Bugün
    if 'bugün' in q:
        weekday = datetime.today().strftime('%A').lower()
        gun_map = {
            'monday': 'pazartesi', 'tuesday': 'salı', 'wednesday': 'çarşamba',
            'thursday': 'perşembe', 'friday': 'cuma'
        }
        gun = gun_map.get(weekday, '')
        dersler = df[(df['gün1'].str.lower() == gun) | (df['gün2'].str.lower() == gun)]
        return f"Bugün ({gun}) olan dersler:\n" + "\n".join(f"- {d}" for d in dersler['ders_adı'].unique()) if not dersler.empty else "Bugün ders yok."

    # Derslik bilgisi
    if 'derslik' in q or 'nerede' in q:
        return f"{row['ders_adı']} dersi:\n- {row['gün1']}: {row['derslik1']}\n- {row['gün2']}: {row['derslik2']}"

    # Hoca dersleri
    if any(hoca.split()[0].lower() in q for hoca in df['hoca-adı'].dropna()):
      for hoca in df['hoca-adı'].dropna().unique():
        ad = hoca.split()[0].lower()
        if ad in q:
            hoca_dersleri = df[df['hoca-adı'].str.lower().str.contains(ad)]['ders_adı'].unique()
            
            if 'dersleri' in q:
                return f"{hoca} hocanın verdiği dersler:\n" + "\n".join(f"- {d}" for d in hoca_dersleri)
            else:
                return f"{hoca} hocanın verdiği bir ders: {hoca_dersleri[0]}"

    if any(phrase in question for phrase in ["hangi dersleri veriyor", "dersleri nelerdir", "derslerin", "verdiği dersler", "hangi ders", "hoca dersleri"]):

        for name in df['hoca-adı'].unique():
            if name.lower() in question:
                hoca_dersleri = df[df['hoca-adı'].str.lower() == name.lower()]['ders_adı'].tolist()
                if hoca_dersleri:
                    ders_listesi = "\n".join([f"- {ders}" for ders in hoca_dersleri])
                    return f"{name} hocanın verdiği dersler:\n{ders_listesi}"
                else:
                    return f"{name} hocaya ait ders bulunamadı."
        return "Lütfen öğretim üyesinin adını doğru yazdığınızdan emin olun."

    # Belirli sınıfın dersleri
    if "sınıfın dersi" in question or "sınıfın dersleri" in question:
        sinif_match = re.search(r"(\d+)\.\s*sınıf", question)
        if sinif_match:
            sinif = sinif_match.group(1)
            dersler = df[df['sınıf'] == int(sinif)]['ders_adı'].tolist()
            if dersler:
                return f"{sinif}. sınıfın dersleri:\n" + "\n".join([f"- {d}" for d in dersler])
            else:
                return f"{sinif}. sınıfa ait ders bulunamadı."

    # Tarihe göre sınav türü ve ders eşleştirme
    date_match = re.search(r"\d{2}\.\d{2}\.\d{4}", question)
    if date_match:
        date_str = date_match.group()
        matched_rows = df[
            (df['vizetarihi'].astype(str) == date_str) |
            (df['finaltarihi'].astype(str) == date_str) |
            (df['butunlemetarihi'].astype(str) == date_str)
        ]
        if not matched_rows.empty:
            response = f"{date_str} tarihinde yapılan sınav(lar):\n"
            for _, row in matched_rows.iterrows():
                if str(row['vizetarihi']) == date_str:
                    response += f"- {row['ders_adı']} (Vize)\n"
                if str(row['finaltarihi']) == date_str:
                    response += f"- {row['ders_adı']} (Final)\n"
                if str(row['butunlemetarihi']) == date_str:
                    response += f"- {row['ders_adı']} (Bütünleme)\n"
            return response
        else:
            return f"{date_str} tarihinde herhangi bir sınav bulunamadı."
     # Belirli bir hocanın verdiği dersleri listeleme
    if any(phrase in question for phrase in ["hangi dersleri veriyor", "dersleri nelerdir", "derslerin", "verdiği dersler", "hangi ders", "hoca dersleri"]):
        question_words = question.split()
        found_hoca = None

        for name in df['hoca-adı'].dropna().unique():
            name_lower = name.lower()
            if name_lower in question:
                found_hoca = name
                break
            name_parts = name_lower.split()
            if any(q_word in name_parts for q_word in question_words):
                found_hoca = name
                break

        if found_hoca:
            # Tüm benzer kayıtları filtrele
            matched_dersler = df[df['hoca-adı'].str.lower().str.contains(found_hoca.lower())]['ders_adı'].tolist()
            if matched_dersler:
                ders_listesi = "\n".join([f"- {ders}" for ders in matched_dersler])
                return f"{found_hoca} hocanın verdiği dersler:\n{ders_listesi}"
            else:
                return f"{found_hoca} hocaya ait ders bulunamadı."
        else:
            return "Hangi hocayı sorduğunuzu anlayamadım. Lütfen tam ad ya da ad-soyad girin."

    # Diğer bilgiler
    if "hoca" in question or "kim" in question:
        return f"Hoca: {match_row['hoca-adı']}"
    elif "sınıf" in question:
        return f"Sınıf: {match_row['sınıf']}"
    elif "vize" in question:
        return f"Vize Tarihi: {match_row['vizetarihi']} Saat: {match_row['saat.1']}"
    elif "final" in question:
        return f"Final Tarihi: {match_row['finaltarihi']} Saat: {match_row['saat.2']}"
    elif "bütünleme" in question:
        return f"Bütünleme Tarihi: {match_row['butunlemetarihi']} Saat: {match_row['saat.3']}"
    elif "gün1" in question or "ilk gün" in question:
        return f"Gün1: {match_row['gün1']} Saat1: {match_row['saat1']} Derslik1: {match_row['derslik1']}"
    elif "gün2" in question or "ikinci gün" in question:
        return f"Gün2: {match_row['gün2']} Saat2: {match_row['saat2']} Derslik2: {match_row['derslik2']}"
    elif "ders" in question:
        return f"Ders: {match_row['ders_adı']}"
    else:
        return "İlgili bilgi belirlenemedi. Tüm detaylar gösteriliyor:"
    # Varsayılan
    return f"{row['ders_adı']} dersi hakkında bilgi: {row['sınıf']}. sınıf, Vize: {row['vizetarihi']}, Final: {row['finaltarihi']}"

# Streamlit Arayüzü
st.title("📚 Ders Bilgi Chatbotu")

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.chat_input("Sorunuzu yazınız...")

if user_input:
    st.session_state.history.append(("user", user_input))
    match_row = find_best_match(user_input)
    response = extract_info(user_input, match_row)
    st.session_state.history.append(("assistant", response))

for role, message in st.session_state.history:
    with st.chat_message(role):
        st.markdown(message)
