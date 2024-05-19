from flask import Flask, request, render_template
import joblib
import PyPDF2
import re

app = Flask(__name__)

# Daha önce kaydedilen modeli ve diğer bileşenleri yükleme
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
label_encoder = joblib.load('label_encoder.pkl')

def cleanResume(resumeText):
    resumeText = re.sub(r'http\S+\s*', ' ', resumeText)  # URL'leri kaldır
    resumeText = re.sub(r'RT|cc', ' ', resumeText)  # RT ve cc'leri kaldır
    resumeText = re.sub(r'#\S+', '', resumeText)  # Hashtag'leri kaldır
    resumeText = re.sub(r'@\S+', '  ', resumeText)  # Kullanıcı adlarını kaldır
    resumeText = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)  # Noktalama işaretlerini kaldır
    resumeText = re.sub(r'[^\x00-\x7f]', r' ', resumeText)  # ASCII dışı karakterleri kaldır
    resumeText = re.sub(r'\s+', ' ', resumeText)  # Ekstra boşlukları kaldır
    return resumeText

def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = []
        for page in reader.pages:
            text.append(page.extract_text())
        return '\n'.join(text)

@app.route('/')
def upload_file():
    return render_template('upload.html')

@app.route('/uploader', methods=['POST'])
def uploader_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        file_path = '/tmp/' + file.filename
        file.save(file_path)
        resume_text = extract_text_from_pdf(file_path)
        cleaned_text = cleanResume(resume_text)
        vectorized_text = vectorizer.transform([cleaned_text])
        prediction = model.predict(vectorized_text)
        category = label_encoder.inverse_transform(prediction)
        
        # Personal information JSON
        personal_info = {
            "parts": {
                "name": "Ceren Ers",
                "email": "ersozceren2@gmail.com",
                "profiles": "www.linkedin.com/in/ceren-ersozhttps://github.com/CerenErsoz",
                "experience": "06/2020 – 06/2022 İzmir, Türkiye\nSoftware Team Member Deu Rov Team\nSoftware that will provide the mobility of the unmanned underwater vehicle. Additionally, processing of images obtained by the vehicle under water.\n09/2022 – 04/2023 İzmir, Türkiye\nSoftware Team Captain Deu Rov Team\nI was the software sub-team captain of the unmanned underwater team.\n20/06/2022 – 22/07/2022 İzmir, Türkiye\nHardware Intern Dokuz Eylül University\nA multidisciplinary hardware internship calculating the e ect of temperature and humidity on textile materials. In this hardware internship, I created hardware using Arduino and DHT11 sensors.\n01/2023 – 02/2023 Ankara, Türkiye\nBackend Developer Intern Cubicl\nI did component development using Node JS, MongoDb, TypeScript, Linux.\n09/12/2023 – 09/01/2024\n.Net Bootcamp Akbank\nDuring the .NET bootcamp, I developed projects using technologies such as Entity Framework, RESTful API, DI Container, Web API, SQL, Redis, Docker. Additionally, I gained in-depth knowledge in development areas such as SOLID principles, design patterns, migration, and middleware usage. I acquired these experiences through the .NET bootcamp organized by Akbank.",
                "education": "AND TRAINING\n15/09/2019 – CURRENT İzmir, Türkiye\nComputer Engineer Dokuz Eylul University\nLANGUAGE SKILLS\nMOTHER TONGUE(S): Turkish\nOTHER LANGUAGE(S): English (Upper intermediate)",
                "skills": "MOTHER TONGUE(S): Turkish\nOTHER LANGUAGE(S): English (Upper intermediate)"
            }
        }

        return render_template('result.html', result=category[0], personal_info=personal_info)

if __name__ == '__main__':
    app.run(debug=True)
