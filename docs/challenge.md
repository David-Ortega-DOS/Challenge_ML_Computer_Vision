
## Configuración del Enterno de trabajo

1. Se utiliza WSL(Windows Subsystem for Linux) para asegurar la compatibilidad de las herramientas solicitadas (make, pythorch, NumPy).
2. Se crearon entornos virtuales aislados y las dependencias se instalaron correctamente utilizando make install dentro del entorno activado (source .venv/bin/activate).
3. Se configuró SSH para mi cuenta de GitHub de David-Ortega-DOS para garantizar pushes seguros y persistentes.


## Parte I: Análisis Exploratorio de Datos (EDA)

A. Análisis de Distribución de Clases y Áreas 

1. Se encontró mayor cantidad de objetos de la clase "forklift" y "person" (con 24219 y 20480 objetos respectivamente), comparados a las demas clases. Este desbalance de clases permite elegir un modelo de entreamiento.
2. Sobre las etiquetas: En Promedio hay 2.4 etiquetas por imagen, hay una o varias imagenes que contiene 47 etiquetas (se analizará en la siguiente sección.). 
3. Sobre el área de las etiquetas: El 50% tienen un área igual o menor al 4.5% del tamaño total de la imagen. Lo que indica que la mayoría de los objetos son pequeños o distantes. Hay auna o varias etiquetas que cubren el 100% del area de la imagen (se analizará en la siguiente sección.)

B. Verificación de la Fiabilidad de las Etiquetas

1. Para verificar la fiabilidad de las etiquetas, se utilizó el Análisis de Outliers (IQR) como método estadístico
2. Sobre el área de las etiquetas: 
    * Se detectaron 5139 (10.6%) de Outliers, al ser un número alto se decidió por analizar otra condición para eliminar.
    * Se detectaron 187 etiquetas con un área del 100% de la imagen, estas se retiraron. 
3. Sobre las etiquetas:
    * Se detectaron 864 (4.21%) de Outliers, al ser una cantidad baja se decidió eliminar todos los Outliers.