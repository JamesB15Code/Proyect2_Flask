from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import pandas as pd
import logging

app = Flask(__name__)

# Configurar el registro
logging.basicConfig(level=logging.DEBUG)

# Cargar el modelo entrenado
model = tf.keras.models.load_model('modelo_bicicletas.h5')
app.logger.debug('Modelo cargado correctamente.')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos enviados en el request
        temp = float(request.form['temp'])
        atemp = float(request.form['atemp'])
        hum = float(request.form['hum'])
        windspeed = float(request.form['windspeed'])
        season = float(request.form['season'])
        
        # Crear un DataFrame con los datos
        data_df = pd.DataFrame([[temp, atemp, hum, windspeed, season]], 
                               columns=['temp', 'atemp', 'hum', 'windspeed', 'season'])
        app.logger.debug(f'DataFrame creado: {data_df}')
        
        # Realizar predicciones
        prediction = model.predict(data_df)
        app.logger.debug(f'Predicción: {prediction[0]}')
        
        # Convertir la predicción a float para que sea serializable a JSON
        prediction_result = float(prediction[0][0])
        
        # Devolver las predicciones como respuesta JSON
        return jsonify({'prediccion': prediction_result})
    except Exception as e:
        app.logger.error(f'Error en la predicción: {str(e)}')
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
