import joblib
import pandas as pd
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import os
from flask_cors import CORS


# --- 1. INICIALIZACIÓN DE LA APP ---
app = Flask(__name__)
CORS(app)

# --- 2. CARGA DE ACTIVOS (MODELO Y PREPROCESADOR) ---
# Construir rutas de forma segura
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'model_files', 'modelo_cancer.h5')
preprocessor_path = os.path.join(base_dir, 'model_files', 'preprocessor.pkl')

print(f"Cargando modelo desde: {model_path}")
model = load_model(model_path)

print(f"Cargando preprocesador desde: {preprocessor_path}")
preprocessor = joblib.load(preprocessor_path)

print("¡Activos cargados! API lista para recibir peticiones.")

# --- 3. DEFINIR LAS COLUMNAS ESPERADAS ---
# ¡CRÍTICO! Deben ser las mismas 13 columnas que usó 'X' en el notebook
# y en el mismo orden.
feature_columns = [
    'age',
    'gender',
    'bmi',
    'alcohol_consumption',
    'smoking_status',
    'hepatitis_b',
    'hepatitis_c',
    'liver_function_score',
    'alpha_fetoprotein_level',
    'cirrhosis_history',
    'family_history_cancer',
    'physical_activity_level',
    'diabetes'
]

# --- 4. ENDPOINT DE PREDICCIÓN ---
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # a. Recibir el JSON de la solicitud
        data = request.get_json()

        # b. Convertir a DataFrame de pandas
        # Aseguramos que las columnas estén en el orden correcto
        patient_df = pd.DataFrame([data], columns=feature_columns)

        # c. Usar el preprocesador para "traducir"
        # ¡IMPORTANTE! Usar .transform(), NO .fit_transform()
        print(f"Datos de entrada (crudos):\n{patient_df}")
        processed_data = preprocessor.transform(patient_df)
        print("Datos procesados listos para el modelo.")

        # d. Usar el modelo para predecir
        # .predict() devuelve un array 2D, ej. [[0.944]]
        probability = model.predict(processed_data)[0][0]
        print(f"Probabilidad (0-1) predicha: {probability}")

        # e. Aplicar lógica de negocio
        riesgo_pct = float(probability * 100) # Convertir a float nativo de Python
        
        if riesgo_pct > 50:
            mensaje = "Alerta: Cita clínica inmediata."
        else:
            mensaje = "Recomendación de seguimiento/chequeos."

        # f. Devolver el resultado en JSON
        response = {
            'riesgo_porcentaje': round(riesgo_pct, 2),
            'mensaje_accion': mensaje
        }
        
        return jsonify(response)

    except Exception as e:
        print(f"Error durante la predicción: {e}")
        return jsonify({'error': str(e)}), 400

# --- 5. EJECUTAR LA APLICACIÓN ---
if __name__ == '__main__':
    # host='0.0.0.0' permite que sea accesible desde el Front-end
    app.run(debug=True, host='0.0.0.0', port=5000)