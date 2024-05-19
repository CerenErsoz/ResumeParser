import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics
import joblib
import warnings

warnings.filterwarnings('ignore')

def cleanResume(resumeText):
    resumeText = re.sub(r'http\S+\s*', ' ', resumeText)  # URL'leri kaldır
    resumeText = re.sub(r'RT|cc', ' ', resumeText)  # RT ve cc'leri kaldır
    resumeText = re.sub(r'#\S+', '', resumeText)  # Hashtag'leri kaldır
    resumeText = re.sub(r'@\S+', '  ', resumeText)  # Kullanıcı adlarını kaldır
    resumeText = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)  # Noktalama işaretlerini kaldır
    resumeText = re.sub(r'[^\x00-\x7f]', r' ', resumeText)  # ASCII dışı karakterleri kaldır
    resumeText = re.sub(r'\s+', ' ', resumeText)  # Ekstra boşlukları kaldır
    return resumeText

# Veri setini yükleme
resumeDataSet = pd.read_csv('UpdatedResumeDataSet.csv', encoding='utf-8')
resumeDataSet['cleaned_resume'] = resumeDataSet['Resume'].apply(lambda x: cleanResume(x))

# Kategori etiketlerini LabelEncoder ile dönüştürme
le = LabelEncoder()
resumeDataSet['Category'] = le.fit_transform(resumeDataSet['Category'])

# Özellik vektörlerini ve hedef değişkeni oluşturma
required_text = resumeDataSet['cleaned_resume'].values
required_target = resumeDataSet['Category'].values

# TF-IDF vektörleri oluşturma
word_vectorizer = TfidfVectorizer(sublinear_tf=True, stop_words='english')
word_vectorizer.fit(required_text)
WordFeatures = word_vectorizer.transform(required_text)

# Eğitim ve test setlerini oluşturma
X_train, X_test, y_train, y_test = train_test_split(WordFeatures, required_target, random_state=42, test_size=0.2, shuffle=True, stratify=required_target)

# KNeighborsClassifier modelini kullanarak eğitim
clf = OneVsRestClassifier(KNeighborsClassifier())
clf.fit(X_train, y_train)

# Modeli kaydetme
joblib.dump(clf, 'model.pkl')
joblib.dump(word_vectorizer, 'vectorizer.pkl')
joblib.dump(le, 'label_encoder.pkl')

# Tahminler yapma ve doğruluk hesaplama
prediction = clf.predict(X_test)
print('Accuracy of KNeighbors Classifier on training set: {:.2f}'.format(clf.score(X_train, y_train)))
print('Accuracy of KNeighbors Classifier on test set: {:.2f}'.format(clf.score(X_test, y_test)))

# Sınıflandırma raporunu görüntüleme
print("\nClassification report for classifier %s:\n%s\n" % (clf, metrics.classification_report(y_test, prediction)))
