import os
import datetime
import logging
from flask import Flask
import mysql.connector
from app.views import *
from app.prediction import *
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.utils.validation")
db_config = {
    'user': 'root',
    'password': 'admin',
    'host': 'localhost',
    'database': 'db_melon',
    'port': '3306'  
}

def connect_to_database():
    try:
        conn = mysql.connector.connect(**db_config)
        print("Connected to MySQL database")
        return conn
    except mysql.connector.Error as err:
        print("Error connecting to MySQL:", err)
        return None
    
app = Flask(__name__)
app.config['uploads'] = os.path.join(os.getcwd(), 'uploads')

db = connect_to_database()
app.config['UPLOAD_FOLDER'] = 'uploads'

def insert_image_info(filename, longitude, latitude):
    conn = connect_to_database()
    if conn:
        try:
            cursor = conn.cursor()
            query = "INSERT INTO images (filename, longitude, latitude) VALUES (%s, %s, %s)"
            cursor.execute(query, (filename, longitude, latitude))
            conn.commit()
            cursor.close()
            conn.close()
            print("Image information inserted successfully")
            return True
        except Exception as e:
            print("Error inserting image information:", e)
            return False
    else:
        return False
    
@app.route('/upload', methods=['POST'])
def upload_image():
    print("Request files:", request.files)  # Debugging statement
    print("Request form:", request.form)    # Debugging statement
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        if not os.path.exists(app.config['uploads']):
            os.makedirs(app.config['uploads'])

        # Simpan file ke folder 'uploads'
        file.save(os.path.join(app.config['uploads'], file.filename))

        # Menerima longitude dan latitude dari request
        longitude = request.form.get('longitude', type=float)
        latitude = request.form.get('latitude', type=float)

        if longitude is None or latitude is None:
            return jsonify({'error': 'Invalid or missing longitude/latitude'}), 400

        # Simpan informasi gambar ke dalam database
        if insert_image_info(file.filename, longitude, latitude):
            return jsonify({'message': 'Image uploaded successfully'}), 200
        else:
            return jsonify({'error': 'Failed to insert image information to database'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    
def label_prediction(prediction):
    if prediction == 1:
        return 'Belum matang'
    elif prediction == 0:
        return 'Siap panen'
    else:
        return 'Tidak Dapat diprediksi'
    
def get_last_uploaded_image():
    conn = connect_to_database()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT filename FROM images ORDER BY id DESC LIMIT 1")
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            if result:
                return result[0]
            else:
                return None
        except Exception as e:
            print("Error fetching last uploaded image:", e)
            return None
    else:
        return None

import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

@app.route('/predict', methods=['GET'])
def predict():
    logging.info('Prediction request received.')
    
    last_uploaded_image = get_last_uploaded_image()
    if last_uploaded_image:
        last_uploaded_image_path = os.path.join(app.config['uploads'], last_uploaded_image)
        logging.info(f'Last uploaded image found: {last_uploaded_image_path}')
        
        try:
            # Menggabungkan dan normalisasi fitur
            combined_features = combine_features(last_uploaded_image_path)
            logging.info('Features combined successfully.')
            
            normalized_features = normalize_features(combined_features, mean_std)
            logging.info('Features normalized successfully.')
            
            # Melakukan prediksi dengan SVM
            prediction_svm = predict_svm(normalized_features)
            logging.info('SVM prediction completed.')
            
            # Melakukan prediksi dengan Random Forest
            prediction_rf = predict_rf(normalized_features)
            logging.info('Random Forest prediction completed.')
            
            # Mengubah prediksi menjadi label
            prediction_label_svm = "Belum matang" if prediction_svm == 1 else "Siap panen"
            prediction_label_rf = "Belum matang" if prediction_rf == 1 else "Siap panen"
            
            # Mengonversi DataFrame ke kamus tanpa menggunakan indeks sebagai kunci
            combined_features_dict = combined_features.to_dict(orient='records')[0]
            normalized_features_dict = normalized_features.to_dict(orient='records')[0]
            
            # Menyimpan hasil prediksi ke dalam database
            try:
                conn = connect_to_database()
                cursor = conn.cursor()
                logging.info('Connected to database.')
                
                # Mendapatkan waktu selesai prediksi
                prediction_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                # Menyimpan setiap fitur ke dalam baris terpisah dalam tabel di database
                query = """INSERT INTO prediction_results 
                           (filename, prediction, prediction_time, contrast, correlation, dissimilarity, energy, homogeneity, jumlah_piksel_jala, kepadatan_piksel_jala) 
                           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
                cursor.execute(query, (
                    last_uploaded_image, 
                    prediction_label_svm, 
                    prediction_time, 
                    normalized_features_dict['contrast'], 
                    normalized_features_dict['correlation'], 
                    normalized_features_dict['dissimilarity'], 
                    normalized_features_dict['energy'], 
                    normalized_features_dict['homogeneity'], 
                    normalized_features_dict['jumlah piksel jala'], 
                    normalized_features_dict['kepadatan piksel jala']))
                conn.commit()
                cursor.close()
                conn.close()
                logging.info('Prediction results saved to database.')
            except Exception as e:
                logging.error(f'Error saving prediction results to database: {e}')
                return jsonify({'error': str(e)})
            
            # Menyusun output menjadi nested
            output = {
                "GLCM":{ "Before normalized": combined_features_dict,
                         "Normalized": normalized_features_dict
                },
                "Prediction_SVM": prediction_label_svm,
                "Prediction_RF": prediction_label_rf,
                'image_filename': last_uploaded_image
            }
            logging.info('Prediction process completed successfully.')
            return jsonify(output)
        except Exception as e:
            logging.error(f'Error during prediction process: {e}')
            return jsonify({'error': str(e)})
    else:
        logging.warning('No uploaded images found.')
        return jsonify({'error': 'No uploaded images found'})
    



if __name__ == "__main__":
    app.run(debug=True, port=8000)