# Backend - Predicción de Riesgo de Cáncer

Este backend proporciona un endpoint REST `/predict` que carga un modelo Keras (`modelo_cancer.h5`) y un `scaler.pkl` para preprocesamiento y devuelve la probabilidad de riesgo.

Requisitos mínimos:

- Python 3.10+ (recomendado). Ten en cuenta la compatibilidad de TensorFlow con la versión de Python.

Pasos para ejecutar localmente (Windows - PowerShell):

1. Abrir PowerShell (recomendado como Administrador si tienes problemas de permisos al instalar paquetes).

2. Ir al directorio `backend`:

```powershell
cd "c:\Users\vcalvarez\Desktop\Sem Sep-Ene 25-26\IA\Proyecto_Inteligencia_Artificial\backend"
```

3. Crear y activar un virtualenv (recomendado):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

4. Actualizar pip e instalar dependencias mínimas:

```powershell
python -m pip install --upgrade pip
python -m pip install flask flask-cors joblib pandas numpy requests scikit-learn h5py
```

5. Si necesitas TensorFlow (para cargar `modelo_cancer.h5`), instala una versión compatible con tu Python:

```powershell
python -m pip install tensorflow==2.16.1
```

Nota: la instalación de TensorFlow puede fallar si la versión de Python no es compatible; en ese caso revisa `python --version` y elige la versión de TensorFlow adecuada.

6. Ejecutar la aplicación:

```powershell
python app.py
```

7. Probar el endpoint con el script incluido o con `Invoke-RestMethod` (PowerShell):

```powershell
# Usando el script de prueba
python test_predict.py

# O usando Invoke-RestMethod
$body = @{
  age = 50
  gender = "Male"
  bmi = 25.0
  alcohol_consumption = "Occasional"
  smoking_status = "Never"
  hepatitis_b = 0
  hepatitis_c = 0
  liver_function_score = 60.0
  alpha_fetoprotein_level = 5.0
  cirrhosis_history = 0
  family_history_cancer = 0
  physical_activity_level = "Moderate"
  diabetes = 0
} | ConvertTo-Json

Invoke-RestMethod -Method Post -Uri http://127.0.0.1:5000/predict -Body $body -ContentType 'application/json'
```

Errores comunes y soluciones:

- Problemas de permisos al instalar paquetes en Windows: cierra editores que puedan bloquear archivos (VSCode), ejecuta PowerShell como administrador o crea un virtualenv y usa ese entorno.
- Error al cargar el modelo: si ves errores relacionados con TensorFlow/h5py, instala una versión de TensorFlow compatible con tu Python.
- Si el `scaler.pkl` fue creado como parte de un pipeline más complejo (ColumnTransformer, OneHotEncoder), asegúrate de guardar el pipeline entero y cargarlo en vez de un scaler aislado.

Si quieres, puedo:

- ajustar `_encode_row` para usar exactamente el preprocesamiento original (necesito el notebook o el pipeline); o
- generar `pipeline.pkl` desde el notebook y actualizar el backend para cargarlo.

---

Backend implementado por el equipo. Si quieres que automatice tests o cree un `Dockerfile` para facilitar despliegue, lo hago a continuación.
