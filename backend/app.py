from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import joblib
import pandas as pd
import numpy as np
# Import pesado de TensorFlow se realiza al cargar el modelo para evitar fallos si
# TensorFlow no está instalado en el entorno de desarrollo o en el analizador.
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, 'model_files', 'modelo_cancer.h5')
SCALER_PATH = os.path.join(BASE_DIR, 'model_files', 'scaler.pkl')

app = Flask(__name__)
CORS(app)


def load_assets():
    """Carga el modelo y el scaler al iniciar la app.
    Devuelve (model, scaler) o lanza excepción si falla.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f'Modelo no encontrado en {MODEL_PATH}')
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f'Scaler no encontrado en {SCALER_PATH}')

    try:
        from tensorflow.keras.models import load_model
    except ImportError as imp_e:
        raise ImportError("TensorFlow no está instalado en el entorno. Instálalo para cargar el modelo: pip install tensorflow") from imp_e

    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


# Cargar activos una vez
MODEL, SCALER = load_assets()

# Columnas esperadas por la API (basado en el CSV de entrenamiento)
EXPECTED_FEATURES = [
    "age",
    "gender",
    "bmi",
    "alcohol_consumption",
    "smoking_status",
    "hepatitis_b",
    "hepatitis_c",
    "liver_function_score",
    "alpha_fetoprotein_level",
    "cirrhosis_history",
    "family_history_cancer",
    "physical_activity_level",
    "diabetes",
]


def _encode_row(data: dict) -> pd.DataFrame:
    """Convierte el JSON de entrada en un DataFrame con las columnas en el orden esperado.
    Esta función aplica mapeos simples a variables categóricas. Si el entrenamiento usó otras
    transformaciones (one-hot, etc.), puede requerir adaptación.
    """
    # Validación básica: rellenar claves faltantes con None
    row = {k: data.get(k, None) for k in EXPECTED_FEATURES}

    # Mapeos para categorías (asumidos; documentar en README si los cambian)
    gender_map = {"Male": 0, "Female": 1, "M": 0, "F": 1}
    alcohol_map = {"Never": 0, "Occasional": 1, "Regular": 2}
    smoking_map = {"Never": 0, "Former": 1, "Current": 2}
    activity_map = {"Low": 0, "Moderate": 1, "High": 2}

    # Aplicar mapeos y normalizar booleanos
    if row["gender"] is not None:
        row["gender"] = gender_map.get(str(row["gender"]).strip(), row["gender"])  # keep original if unknown

    if row["alcohol_consumption"] is not None:
        row["alcohol_consumption"] = alcohol_map.get(str(row["alcohol_consumption"]).strip(), row["alcohol_consumption"]) 

    if row["smoking_status"] is not None:
        row["smoking_status"] = smoking_map.get(str(row["smoking_status"]).strip(), row["smoking_status"]) 

    if row["physical_activity_level"] is not None:
        row["physical_activity_level"] = activity_map.get(str(row["physical_activity_level"]).strip(), row["physical_activity_level"]) 

    # Asegurar tipos numéricos para campos binarios/enteros
    for col in ["hepatitis_b", "hepatitis_c", "cirrhosis_history", "family_history_cancer", "diabetes"]:
        val = row.get(col)
        if isinstance(val, bool):
            row[col] = int(val)
        elif val is None or val == "":
            row[col] = np.nan
        else:
            try:
                row[col] = int(val)
            except (ValueError, TypeError):
                row[col] = np.nan

    # Convertir numéricos
    for col in ["age", "bmi", "liver_function_score", "alpha_fetoprotein_level", "gender", "alcohol_consumption", "smoking_status", "physical_activity_level"]:
        val = row.get(col)
        if val is None or val == "":
            row[col] = np.nan
        else:
            try:
                row[col] = float(val)
            except (ValueError, TypeError):
                # si no puede convertirse, dejar como NaN para manejo posterior
                row[col] = np.nan

    df = pd.DataFrame([row], columns=EXPECTED_FEATURES)
    return df


@app.route('/predict', methods=['POST'])
def predict():
    payload = request.get_json(force=True, silent=True)
    if not payload:
        return jsonify({"error": "JSON body requerido"}), 400

    # validación de campos mínimos
    missing = [f for f in EXPECTED_FEATURES if f not in payload]
    if missing:
        # no abortamos si faltan (para permitir defaults), pero informamos
        return jsonify({"error": "Faltan campos en la petición", "missing": missing}), 400

    df = _encode_row(payload)

    # Aplicar el mismo preprocesamiento (scaler). Intentamos varias estrategias robustas.
    try:
        # Muchos scaler/transformers aceptan DataFrame directamente
        X_proc = SCALER.transform(df)
    except (AttributeError, ValueError, TypeError):
        # Intentar escalar sólo columnas numéricas conocidas
        numeric_cols = ["age", "bmi", "liver_function_score", "alpha_fetoprotein_level"]
        try:
            nums = df[numeric_cols].astype(float)
            nums_scaled = SCALER.transform(nums)
            # reconstruir entrada concatenando el resto
            rest_cols = [c for c in df.columns if c not in numeric_cols]
            X_proc = np.hstack([nums_scaled, df[rest_cols].fillna(0).values.astype(float)])
        except (KeyError, ValueError, TypeError) as e2:
            return jsonify({"error": "No se pudo aplicar el scaler al input", "detail": str(e2)}), 500

    try:
        proba = MODEL.predict(X_proc)
    except (ValueError, RuntimeError, TypeError) as e:
        # Los errores de predict pueden ser variados (shape mismatch, modelo corrupto, etc.)
        return jsonify({"error": "Error al ejecutar la predicción", "detail": str(e)}), 500

    # si devuelve array [[p]]
    try:
        if isinstance(proba, (list, np.ndarray)):
            proba_val = float(np.asarray(proba).ravel()[0])
        else:
            proba_val = float(proba)
    except (TypeError, ValueError) as e:
        return jsonify({"error": "Formato inesperado en la salida del modelo", "detail": str(e)}), 500

    percent = round(proba_val * 100, 2)
    if percent <= 50:
        message = "Recomendación de seguimiento/chequeos."
    else:
        message = "Alerta: Cita clínica inmediata."

    return jsonify({"risk_percent": percent, "message": message, "probability": proba_val})


if __name__ == '__main__':
    # Ejecutar en modo desarrollo en el puerto 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
