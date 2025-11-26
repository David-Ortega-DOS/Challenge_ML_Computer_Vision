# Cuaderno de Trabajo: Machine Learning Computer Vision Challenge

## Resumen Ejecutivo

Este documento detalla el desarrollo e implementación de un modelo de Detección de Objetos YOLO11m para identificar 17 clases en entornos logísticos. Tras la limpieza de datos mediante manejo de outliers y luego de migrar el modelo desde el nano al medium, se logró mejorar el rendimiento del modelo inicial en un 53% (mAP50-95 = 0.197). La solución fue empaquetada en una API FastAPI y desplegada en Google Cloud Run mediante un pipeline automatizado de CI/CD con GitHub Actions. El sistema pasa satisfactoriamente las pruebas de Recall y estabilidad del servicio.

### Accesos rápidos:

* https://fastapi-yolo-service-466705638168.us-central1.run.app/docs
* https://github.com/David-Ortega-DOS/Challenge_ML_Computer_Vision 

## Configuración del Entorno de trabajo

1.  **Plataforma Base:** Se utilizó WSL (Windows Subsystem for Linux) para garantizar la compatibilidad de herramientas de ML (make, PyTorch, NumPy).
2.  **Aislamiento:** Se crearon entornos virtuales aislados y las dependencias se instalaron utilizando `make install`.
3.  **Control de Versiones:** Se configuró SSH para mi cuenta de GitHub (`David-Ortega-DOS`) para asegurar *pushes* seguros y la trazabilidad del código.
4.  **Entrenamiento** Para el entrenamiento se usó un Entorno de Ejecución de Google Colab de 160GB de RAM, 80GB de GPU y 235 GB de Disco para optimizar el tiempo de ejecución.

---

# Parte I: Desarrollo y Evaluación del Modelo

## 1. Análisis Exploratorio de Datos (EDA)

### A. Análisis de Distribución de Clases y Áreas 

1. **Desbalance de Clases:** Se encontró mayor cantidad de objetos de la clase "forklift" y "person" (con 24219 y 20480 objetos respectivamente), comparados a las demas clases. Este desbalance de clases nos indica que se debe escoger un número de "EPOCHs" mas alto del promedio para dar tiempo a las clases minoritarias a converger.
2. **Densidad de Etiquetas:** En Promedio hay 2.4 etiquetas por imagen, Se identificaron imágenes con 47 etiquetas, indicando la presencia de escenas complejas y altamente pobladas. 
3. **Tamaño de Objeto:** El 50%  de los objetos tienen un área igual o menor al 4.5% del tamaño total de la imagen. Lo que confirma que la mayoría de los objetos son pequeños o distantes. 

### B. Verificación de la Fiabilidad de las Etiquetas

1. Para verificar la fiabilidad de las etiquetas, se utilizó el Análisis de Outliers (IQR) 
2. Sobre el tamaño de objeto: 
    * Se detectaron 5139 (10.6%) de Outliers, al ser un número alto se decidió por analizar otra condición para eliminar.
    * Se detectaron 187 etiquetas con un área del 100% de la imagen, estas se retiraron. 
3. Sobre la densidad de etiquetas:
    * Se detectaron 864 (4.21%) de Outliers, al ser una cantidad baja se decidió eliminar todos los Outliers.

---

## 2. Training

Para el entrenamiento se usó un Entorno de Ejecución de Google Colab de 160GB de RAM, 80GB de GPU y 235 GB de Disco.

### A. HYPERPARAMETERS

| Entrenamiento | Modelo Base | EPOCHS | IMGSZ | BATCH | mAP50 | mAP50-95 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Primero** | $\text{YOLO11n}$ | 40 | 800 | 64 | 0.192 | 0.128 |
| **Segundo (Final)** | $\text{YOLO11m}$ | 40 | 896 | 80 | 0.282 | 0.197 |


### B. Discussion

1. **Why did you choose these hyperparameters?**  
    * EPOCHS = 40: Se eligieron 40 épocas para garantizar una convergencia suficiente para las clases minoritarias
    * IMGSZ  = 896: Se aumentó la resolución de la imagen de entrada de 640 a 896 para ayudar al modelo a identificar y localizar con mayor precisión los objetos pequeños y distantes (la mediana del área de las bounding boxes era muy baja (15,5%))
    * BATCH  = 80: Se eligió un batch size muy grande (80) para aprovechar al máximo la 80GB de VRAM de la GPU A100 y reducir drásticamente el número de iteraciones totales

2. **How do they affect training time, GPU/CPU usage, and accuracy?** 
    * Training time: Al aumentar el batch size, el número de pasos de backpropagation se reduce, lo que minimiza el tiempo total.
    * GPU/CPU usage: El valor alto del BATCH aumenta el uso de GPU, no se modificaron los workers (8 por defecto) lo que aumenta el trabajo en el CPU, pero evita el embotellamiento 
    * Accuracy: La resolución más alta da más información al modelo sobre los objetos pequeños, lo que tiende a mejorar las métricas mAP50 y mAP50-95
    
3. **What would you try differently if you had more time or resources?**
    * Aumentar la Capacidad del Modelo por un modelo yolo11l, para manejar mejor el desbalance de clases.
    * Implementar un over-sampling en las clases con pocos datos.
    * Realizar mas entrenamientos con diferentes hiperparámetros para encontrar el mejor modelo. 

---

## 3. Evaluation

Metrics Interpretation and Analysis, Provide a short written analysis here:

### 1. Quantitative Summary:

* **What are your mAP50 and mAP50-95 values**
    En las métricas globales de mAP50 (**0.282**) y mAP50-95 (**0.197**) se obtuvieron valores de precisión bajos (debido al desbalance de clases), pero que se pueden mejorar más (se mejoraron al migrar de modelo nano a medium). 

* **Which classes achieved the highest and lowest detection performance?**

    - **forklift**: tiene el mejor rendimiento: mAP50-95 = 0.772 y mAP50 = 0.963
    - **van**: tiene el peor rendimiento mAP50-95 = 0.00952 y mAP50 = 0.0119
        
* **Rendimiento por Clase:**

    | Class | Instances | R | mAP50 | mAP50-95 |
    | :--- | :--- | :--- | :--- | :--- |
    | all | 4615 | 0.346 | 0.282 | 0.197 |
    | car | 31 | 0.765 | 0.509 | 0.367 |
    | cardboard box | 64 | 0.188 | 0.17 | 0.127 |
    | **forklift** | **2310** | **0.952** | **0.963** | **0.772** |
    | freight container | 23 | 0.217 | 0.126 | 0.0572 |
    | gloves | 6 | 0.167 | 0.25 | 0.192 |
    | helmet | 22 | 0.182 | 0.216 | 0.149 |
    | ladder | 23 | 0.348 | 0.208 | 0.0956 |
    | license plate | 23 | 0.348 | 0.157 | 0.12 |
    | person | 1938 | 0.872 | 0.846 | 0.545 |
    | road sign | 19 | 0.0912 | 0.0238 | 0.0179 |
    | safety vest | 13 | 0.229 | 0.221 | 0.0825 |
    | traffic cone | 97 | 0.0309 | 0.0679 | 0.0349 |
    | traffic light | 2 | 0 | 0.0192 | 0.00766 |
    | truck | 14 | 0.929 | 0.504 | 0.391 |
    | **van** | **2** | **0** | **0.0119** | **0.00952** |
    | wood pallet | 28 | 0.214 | 0.216 | 0.186 |


### 2. Qualitative Analysis:

* **Describe common failure cases:**
    - **Clases Omitidas (Recall Bajo)**: El modelo no logra predecir ninguna instancia las clases "traffic light" y "van"
    - **Objetos Pequeños Perdidos**: Dado que la mediana del área de las bounding boxes era baja  y que solo se logró un mAP50-95 = 0.197, se concluye que el modelo está fallando consistentemente en detectar objetos pequeños y distantes aun con el aumento de resolución (Recall cercano a cero para "road sing" y "traffic cone")

* **Were there any label quality issues or inconsistencies you observed?**
    - Se identificó un 10.60% de outliers de área en el EDA. Aunque se eliminaron las etiquetas extremas (área=1) por ser ruido (objetos truncados), el alto porcentaje restante de variabilidad de escala es una inconsistencia inherente que el modelo no pudo manejar con éxito.

### 3. Improvement Proposals:
* **Suggest at least two improvements:**.
    - Migracion de modelo por un modelo yolo11l, para mejorar el balanceo de clases.
    - Aplicar pesos de clase (class_weights) o técnicas de oversampling para las clases de menor instancias.


* **How would you validate whether these changes actually help?**
    - Se comprobaría si el mAP50-95 duplica su valor, para justificar el costo computacional adicional.
    - Se reentrenaría el modelo con los pesos ajustados. La validación se centraría en el aumento del mAP50-95 individual de las clases minoritarias. 

---

# Parte II: Implementación de la API (FastAPI)

La API se implementó con **FastAPI** (`challenge/api.py`). El modelo best.pt se carga con prioridad sobre el ONNX para garantizar la estabilidad en la CPU del *runner* de CI.

| Parámetro Clave | Valor |
| :--- | :--- |
| **Prioridad de Carga** | PT > ONNX |
| **Umbral de Confianza (CI)** | 0.25 |
| **Umbral de IoU (CI)** | .0.5 |
| **Rutas** | Absolutas (`ROOT_DIR / "artifacts" / "model.onnx"`) |

![Gráfico del make api-test exitoso](https://github.com/David-Ortega-DOS/Challenge_ML_Computer_Vision/blob/main/docs/img/make-api-test.png)

---

# Parte III: Despliegue en Cloud Run

El servicio se empaquetó con una imagen base optimizada (`tiangolo/uvicorn-gunicorn-fastapi:python3.10`) y se desplegó en Google Cloud Run.

* **URL del Servicio Final:** `https://fastapi-yolo-service-466705638168.us-central1.run.app/docs`

![Gráfico del servicio deplegado](https://github.com/David-Ortega-DOS/Challenge_ML_Computer_Vision/blob/main/docs/img/fastapi-yolo-service.png)

---

# Parte IV: Implementación de CI/CD (GitHub Actions)

Se implementó el **pipeline** de CI/CD para automatizar pruebas y despliegue.

## 1. Configuración de Automatización

* **Autenticación:** Se utilizó GCP Service Account Key con roles de Cloud Run Admin y Artifact Registry Writer.
* **CI (`ci.yml`):** Ejecuta make api-test y valida que Recall > 0.10.
* **CD (`cd.yml`):** Construye la imagen, la sube a GCR y despliega en Cloud Run.

## 2. Resultado Final del Pipeline

* **CI Status:** PASSED  $\checkmark$. El Recall fue satisfactorio gracias al desempeño del modelo .pt.
* **CD Status:** COMPLETED $\checkmark$. La versión final del código está en producción. 

![Gráfico de los workflows exitosos en GitHub Actions CI/CD ](https://github.com/David-Ortega-DOS/Challenge_ML_Computer_Vision/blob/main/docs/img/github_actions.png)

--

# Prueba de Ejecución 

1. En la siguente imagen se muestra como el API reconoce exitosamente los marcadores correspondientes a "forklift" y "person"

![Gráfico del post predic ](https://github.com/David-Ortega-DOS/Challenge_ML_Computer_Vision/blob/main/docs/img/post_predict.png)

2. En el siguiente video (hacer click en la imagen) se muestra como se realiza la ejecución del API desplejada. Se obtiene el marcador del "forklift" que se encontraba en la imagen.

[![Vista Previa del Video 1](https://github.com/David-Ortega-DOS/Challenge_ML_Computer_Vision/blob/main/docs/img/video1.png)](https://drive.google.com/file/d/1xXk-sySnZm4y49EsCp1UZmXdoHEo1BPN/view?usp=drive_link)


3. En este video (hacer click en la imagen) se prueba con otra imagen y se obtienen los marcadores correspondientes a "forklift" y "person" de manera correcta. 

[![Vista Previa del Video 2](https://github.com/David-Ortega-DOS/Challenge_ML_Computer_Vision/blob/main/docs/img/video2.png)](https://drive.google.com/file/d/1sCwvdKYmIZpM4NqKLhucwhhW_cN3qEx9/view?usp=drive_link)