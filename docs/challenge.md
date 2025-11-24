
# Configuración del Enterno de trabajo

1. Se utiliza WSL(Windows Subsystem for Linux) para asegurar la compatibilidad de las herramientas solicitadas (make, pythorch, NumPy).
2. Se crearon entornos virtuales aislados y las dependencias se instalaron correctamente utilizando make install dentro del entorno activado (source .venv/bin/activate).
3. Se configuró SSH para mi cuenta de GitHub de David-Ortega-DOS para garantizar pushes seguros y persistentes.


# Parte I: 

## 1. Análisis Exploratorio de Datos (EDA)

A. Análisis de Distribución de Clases y Áreas 

1. Se encontró mayor cantidad de objetos de la clase "forklift" y "person" (con 24219 y 20480 objetos respectivamente), comparados a las demas clases. Este desbalance de clases nos indica que se debe escoger un número de "EPOCHs" mas alto del promedio.
2. Sobre las etiquetas: En Promedio hay 2.4 etiquetas por imagen, hay una o varias imagenes que contiene 47 etiquetas (se analizará en la siguiente sección.). 
3. Sobre el área de las etiquetas: El 50% tienen un área igual o menor al 4.5% del tamaño total de la imagen. Lo que indica que la mayoría de los objetos son pequeños o distantes. Hay auna o varias etiquetas que cubren el 100% del area de la imagen (se analizará en la siguiente sección.)

B. Verificación de la Fiabilidad de las Etiquetas

1. Para verificar la fiabilidad de las etiquetas, se utilizó el Análisis de Outliers (IQR) como método estadístico
2. Sobre el área de las etiquetas: 
    * Se detectaron 5139 (10.6%) de Outliers, al ser un número alto se decidió por analizar otra condición para eliminar.
    * Se detectaron 187 etiquetas con un área del 100% de la imagen, estas se retiraron. 
3. Sobre las etiquetas:
    * Se detectaron 864 (4.21%) de Outliers, al ser una cantidad baja se decidió eliminar todos los Outliers.


## 2. Training

Para el entrenamiento se usó un Entorno de Ejecución de Google Colab de 160GB de RAM, 80GB de GPU y 235 GB de Disco.

A. HYPERPARAMETERS

    * Primer Entrenamiento: yolo11n.pt --> mAP50 (0.192) y mAP50-95 (0.128)

        EPOCHS = 40   #se aumentaron las epocas de 30 a 40 debido al desbalance en clases
        IMGSZ  = 800  #mayor resolucion de las imagenes debido a objetos pequeños 
        BATCH  = 64   #para reducir el número de iteraciones
        SEED = 42     #para reproductividad
    
    * Segundo Entrenamiento: yolo11m.pt --> mAP50 (0.282) y mAP50-95 (0.197)

        EPOCHS = 40   
        IMGSZ  = 896  #se aumentó para mejorar resolucion de objetos pequeños
        BATCH  = 80   #para reducir el número de iteraciones
        SEED = 42

B. Discussion

1. Why did you choose these hyperparameters?  
    * EPOCHS = 40: Se eligieron 40 épocas para garantizar una convergencia suficiente para las clases minoritarias
    * IMGSZ  = 896: Se aumentó la resolución de la imagen de entrada de 640 a 896 para ayudar al modelo a identificar y localizar con mayor precisión los objetos pequeños y distantes (la mediana del área de las bounding boxes era muy baja (15,5%))
    * BATCH  = 80: Se eligió un batch size muy grande (80) para aprovechar al máximo la 80GB de VRAM de la GPU A100 y reducir drásticamente el número de iteraciones totales

2. How do they affect training time, GPU/CPU usage, and accuracy?  
    * Training time: Al aumentar el batch size, el número de pasos de backpropagation se reduce, lo que minimiza el tiempo total.
    * GPU/CPU usage: El valor alto del BATCH aumenta el uso de GPU, no se modificaron los workers (8 por defecto) lo que aumenta el trabajo en el CPU, pero evita el embotellamiento 
    * Accuracy: La resolución más alta da más información al modelo sobre los objetos pequeños, lo que tiende a mejorar las métricas mAP50 y mAP50-95
    
3. What would you try differently if you had more time or resources?
    * Aumentar la Capacidad del Modelo por un modelo yolo11m, para manejar mejor el desbalance de clases.
    * Implementar un over-sampling en las clases con pocos datos.
    * Realizar mas entrenamientos con diferentes hiperparámetros para encontrar el mejor modelo. 


## 3. Evaluation

Metrics Interpretation and Analysis, Provide a short written analysis here:

1. Quantitative Summary:

    * What are your mAP50 and mAP50-95 values?
        En las métricas globales de mAP50 (0.282) y mAP50-95 (0.197) se obtuvieron valores de precisión bajos, pero que se pueden mejorar más (se mejoraron al migrar de modelo nano a medium). 

  
    * Which classes achieved the highest and lowest detection performance?

        "forklift" tiene el mejor rendimiento: mAP50-95 = 0.772 y mAP50 = 0.963
        "van" tiene el peor rendimiento mAP50-95 = 0.00952 y mAP50 = 0.0119
        
        En las métricas por clases hay mucha diferencia:
        
                 Class     Images  Instances      Box          R      mAP50      mAP50-95
                   all       1967       4615      0.393      0.346      0.282      0.197
                   car         14         31      0.367      0.765      0.509      0.367
         cardboard box         18         64      0.292      0.188       0.17      0.127
              forklift       1967       2310      0.832      0.952      0.963      0.772
     freight container         13         23      0.307      0.217      0.126     0.0572
                gloves          4          6      0.492      0.167       0.25      0.192
                helmet         15         22      0.411      0.182      0.216      0.149
                ladder         19         23      0.209      0.348      0.208     0.0956
         license plate         21         23      0.177      0.348      0.157       0.12
                person       1029       1938      0.697      0.872      0.846      0.545
             road sign         16         19     0.0758     0.0912     0.0238     0.0179
           safety vest          7         13      0.426      0.229      0.221     0.0825
          traffic cone         21         97      0.226     0.0309     0.0679     0.0349
         traffic light          2          2          0          0     0.0192    0.00766
                 truck         11         14      0.391      0.929      0.504      0.391
                   van          2          2          1          0     0.0119    0.00952
           wood pallet         15         28       0.39      0.214      0.216      0.186


2. Qualitative Analysis:

    * Describe common failure cases (e.g., small objects missed, overlapping detections, background confusion).
        Clases Omitidas (Recall Bajo): El modelo no logra predecir ninguna instancia las clases "traffic light" y "van"
        Objetos Pequeños Perdidos: Dado que la mediana del área de las bounding boxes era baja  y que solo se logró un mAP50-95 = 0.197, se concluye que el modelo está fallando consistentemente en detectar objetos pequeños y distantes aun con el aumento de resolución.

    * Were there any label quality issues or inconsistencies you observed?
        Se identificó un 10.60% de outliers de área en el EDA. Aunque se eliminaron las etiquetas extremas (área=1) por ser ruido (objetos truncados), el alto porcentaje restante de variabilidad de escala es una inconsistencia inherente que el modelo no pudo manejar con éxito.

3. Improvement Proposals:
    * Suggest at least two improvements (data augmentation, loss tuning, class balancing, etc.).
        Migracion de modelo por un modelo yolo11l, para mejorar el balanceo de clases.
        Aplicar pesos de clase (class_weights) o tecnias de oversampling para las clases de menor instancias.


    * How would you validate whether these changes actually help?
        Se comprobaría si el mAP50-95 duplica su valor, para justificar el costo computacional adicional.
        Se reentrenaría el modelo con los pesos ajustados. La validación se centraría en el aumento del mAP50-95 individual de las clases minoritarias. 

