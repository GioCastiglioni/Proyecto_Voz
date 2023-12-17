# Proyecto Voz: Reconocimiento de Emociones en Voz Usando Redes Recurrentes

## Contextualización

El reconocimiento de emociones mediante la voz, en inglés Speech Emotion Recognition (`SER`), representa un avance significativo en la creación de robots sociales capaces de entender y responder a las necesidades humanas. Estos sistemas utilizan patrones vocales para identificar el estado emocional de un usuario, lo cual es crucial para el desarrollo de interacciones hombre-máquina más naturales y efectivas. A diferencia de los métodos que dependen del análisis visual, el `SER` aprovecha la voz humana, la cual es menos susceptible a obstrucciones o variaciones en la visibilidad, como podría ser el caso cuando la cara está parcialmente oculta.

Con el objetivo de mejorar la interacción entre humanos y máquinas, el proyecto a realizar busca implementar un sistema `SER` utilizando técnicas de Deep Learning. Este enfoque se centra en procesar señales de audio para determinar la emoción expresada por el usuario. A través de la extracción de características y la clasificación de emociones, el sistema desarrollado deberá discernir entre diferentes estados afectivos, como felicidad, tristeza o neutralidad.

## Propuesta de Solución

Los features que se utilizarán para caracterizar los audios del dataset a trabajar corresponden a coeficientes LPC y coeficientes MFCC.

Los coeficientes LPC son efectivos para modelar los formantes de la voz, que son concentraciones de energía alrededor de ciertas frecuencias y son cruciales para la identificación de sonidos vocálicos en el habla. Esta característica puede ser sumamente relevante para un sistema `SER`, ya que los formantes pueden variar con las distintas emociones.

Aunque los coeficientes LPC no conservan directamente la información de tono de la voz, son capaces de reconstruir la forma de onda de la señal de habla cuando se complementan con un tren de impulsos.

Por otra parte, los coeficientes MFCC se basan en la escala mel, que está diseñada para imitar la percepción no lineal del oído humano de las frecuencias. Además, los coeficientes MFCC proporcionan una representación del espectro de energía de la voz que captura las características importantes para el reconocimiento de emociones, como las variaciones en la energía y la dinámica de las frecuencias a lo largo del tiempo.

Es por ello que, mientras que los coeficientes LPC son buenos para modelar la forma de la onda de la señal de habla y sus propiedades de resonancia, los coeficientes MFCC son robustos en capturar la textura y la calidad del sonido relacionada con la percepción humana, por lo que la combinación de ambos coeficientes proporciona una mayor información sobre la voz que permitiría identificar emociones en esta.

Respecto a las arquitecturas de modelos a utilizar, se tendrá un mayor enfoque temporal, por lo que los modelos recurrentes son útiles para procesar secuencias de datos en donde la información emocional está distribuida a lo largo de la señal de habla.

## Contenido

En el presente repositorio se encuentran los códigos para extraer los coeficientes LPC y MFCC de los audios del dataset de CREMA-D, así como los archivos .h5 resultantes. También se encuentra el archivo .ipynb correspondiente al informe y código del proyecto, la carpeta "test" que contiene los coeficientes y targets del conjunto de prueba, y la carpeta "results" que contiene los pesos resultantes de los mejores modelos Vanilla RNN y LSTM RNN.
