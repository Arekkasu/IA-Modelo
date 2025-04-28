from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input  # Asegúrate de importar preprocess_input correctamente
import numpy as np
import io
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Habilita CORS para todas las rutas y orígenes

# Cargar el modelo
modelo = load_model('modelo_debe_funcionar.keras')

# Dimensiones de entrada del modelo
img_width, img_height = 384, 384

# Mapeo de clases (debes tener este mapeo según el entrenamiento de tu modelo)
class_indices = {'cardboard': 0, 'glass': 1, 'metal': 2, 'paper': 3, 'plastic': 4, 'trash': 5}  # Ejemplo de mapeo

@app.route('/predecir', methods=['POST'])
def predecir():
    if request.method == 'POST':
        # Obtener la imagen del request
        file = request.files['imagen']

        # Convertir FileStorage a BytesIO
        in_memory_file = io.BytesIO()
        file.save(in_memory_file)
        in_memory_file.seek(0)

        # Cargar la imagen desde BytesIO y normalizar automáticamente
        img = image.load_img(in_memory_file, target_size=(img_width, img_height))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)  # Normalización usando preprocess_input de Keras

        # Hacer la predicción
        prediccion = modelo.predict(x)

        # Obtener la clase con mayor probabilidad
        clase_predicha = np.argmax(prediccion[0])

        # Obtener la etiqueta de la clase
        etiqueta_predicha = list(class_indices.keys())[list(class_indices.values()).index(clase_predicha)]

        # Devolver la predicción como JSON
        return jsonify({'clase': etiqueta_predicha, 'probabilidad': float(prediccion[0][clase_predicha])})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
