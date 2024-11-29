import tensorflow as tf
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
import io

app = Flask(__name__)

# Define las clases generales (posturas)
CLASSES_GENERALES = ["frontal", "posterior", "latDerecho", "latIzquierdo"]

# Ruta del modelo TensorFlow Lite
TFLITE_MODEL_PATH = "model/modelo_postura_general.tflite"

# Cargar el intérprete de TensorFlow Lite
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()

# Obtener detalles de entrada y salida del modelo
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

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
        
        # Redimensionar la imagen al tamaño esperado por el modelo
        imagen = imagen.resize((256, 256))
        
        # Convertir la imagen a un numpy array y normalizar
        imagen_array = np.array(imagen, dtype=np.float32) / 255.0
        
        # Expandir dimensiones para que coincida con la entrada del modelo
        imagen_array = np.expand_dims(imagen_array, axis=0)
        
        return imagen_array
    except Exception as e:
        raise ValueError(f"Error procesando la imagen: {str(e)}")

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Servidor Flask funcionando correctamente',
        'classes': CLASSES_GENERALES
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No se proporcionó ninguna imagen.'}), 400

        imagen = request.files['file'].read()
        input_data = procesar_imagen(imagen)

        # Configurar el tensor de entrada
        interpreter.set_tensor(input_details[0]['index'], input_data)

        # Ejecutar el modelo
        interpreter.invoke()

        # Obtener los resultados de salida
        predictions = interpreter.get_tensor(output_details[0]['index'])

        # Determinar la clase con mayor probabilidad
        predicted_class = np.argmax(predictions[0])
        predicted_posture = CLASSES_GENERALES[predicted_class]

        return jsonify({'prediction': predicted_posture})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print(f"Running on http://0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
