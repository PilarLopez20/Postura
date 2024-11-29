from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)

# Define las clases generales (posturas) manualmente
CLASSES_GENERALES = ["frontal", "posterior", "latDerecho", "latIzquierdo"]

# Carga el modelo directamente desde el archivo .h5
model = tf.keras.models.load_model('model/modelo_postura_general.h5')

def procesar_imagen(imagen_bytes):
    """
    Procesa una imagen para extraer características necesarias para el modelo.

    Args:
        imagen_bytes (bytes): Imagen en formato binario enviada por el usuario.
    
    Returns:
        np.array: Array procesado con 120 coordenadas (normalizadas).
    """
    try:
        # Cargar la imagen con PIL
        imagen = Image.open(io.BytesIO(imagen_bytes)).convert('RGB')
        
        # Redimensionar la imagen a 256x256 (tamaño esperado)
        imagen = imagen.resize((256, 256))
        
        # Convertir la imagen a un numpy array y normalizar
        imagen_array = np.array(imagen) / 255.0
        
        # Flatten (aplanar la imagen) para simular características
        imagen_flatten = imagen_array.flatten()
        
        # Ajustar a 120 coordenadas si es necesario
        if len(imagen_flatten) < 120:
            # Rellenar con ceros si tiene menos de 120 valores
            imagen_flatten = np.pad(imagen_flatten, (0, 120 - len(imagen_flatten)), mode='constant')
        else:
            # Cortar si tiene más de 120 valores
            imagen_flatten = imagen_flatten[:120]
        
        return np.array([imagen_flatten])
    except Exception as e:
        raise ValueError(f"Error procesando la imagen: {str(e)}")

# Ruta principal para comprobar que el servidor está activo
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Servidor Flask funcionando correctamente en Railway',
        'classes': CLASSES_GENERALES
    })

# Ruta para realizar predicciones a partir de imágenes
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Verifica si hay un archivo de imagen en la solicitud
        if 'file' not in request.files:
            return jsonify({'error': 'No se proporcionó ninguna imagen.'}), 400

        # Lee la imagen desde la solicitud
        imagen = request.files['file'].read()

        # Procesa la imagen para extraer características
        input_data = procesar_imagen(imagen)

        # Realiza la predicción
        predictions = model.predict(input_data)

        # Obtiene el índice de la clase con mayor probabilidad
        predicted_class = np.argmax(predictions, axis=1)[0]

        # Traduce el índice a la clase general
        predicted_posture = CLASSES_GENERALES[predicted_class]

        # Devuelve la predicción como JSON
        return jsonify({'prediction': predicted_posture})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Railway asigna el puerto como una variable de entorno
    port = int(os.environ.get("PORT", 5000))  # Usa el puerto 5000 como predeterminado si no se encuentra la variable
    app.run(host='0.0.0.0', port=port, debug=True)
