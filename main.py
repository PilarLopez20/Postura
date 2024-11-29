import os
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Forzar uso de CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Desactiva GPU

app = Flask(__name__)

# Define las clases generales (posturas)
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
            imagen_flatten = np.pad(imagen_flatten, (0, 120 - len(imagen_flatten)), mode='constant')
        else:
            imagen_flatten = imagen_flatten[:120]
        
        return np.array([imagen_flatten])
    except Exception as e:
        raise ValueError(f"Error procesando la imagen: {str(e)}")

# Ruta principal para comprobar que el servidor está activo
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Servidor Flask funcionando correctamente',
        'classes': CLASSES_GENERALES
    })

# Ruta para realizar predicciones a partir de imágenes
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No se proporcionó ninguna imagen.'}), 400

        imagen = request.files['file'].read()
        input_data = procesar_imagen(imagen)
        predictions = model.predict(input_data)
        predicted_class = np.argmax(predictions, axis=1)[0]
        predicted_posture = CLASSES_GENERALES[predicted_class]

        return jsonify({'prediction': predicted_posture})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Render usa la variable de entorno PORT
    print(f"Running on http://0.0.0.0:{port}")  # Diagnóstico
    app.run(host='0.0.0.0', port=port, debug=False)  # debug=False para producción
