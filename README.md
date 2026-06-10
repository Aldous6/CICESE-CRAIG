# CRAIG Estancia

Implementación y evaluación del algoritmo CRAIG (*Coresets for Data-efficient Training of Machine Learning Models*) para selección de subconjuntos de datos orientada a reducir el costo computacional del entrenamiento de modelos de aprendizaje automático.

## Objetivo

Implementar una versión modular en Python de CRAIG y evaluar su desempeño frente a dos baselines:

1. Entrenamiento con todos los datos.
2. Entrenamiento con subconjuntos aleatorios.
3. Entrenamiento con subconjuntos seleccionados mediante CRAIG.

## Primera etapa

La primera implementación se enfocará en clasificación supervisada con regresión logística, usando:

- Selección estratificada por clase.
- Función tipo facility location.
- Algoritmo greedy.
- Pesos gamma para los puntos seleccionados.
- Entrenamiento ponderado con `sample_weight`.

## Métricas

- Accuracy.
- Log loss.
- Tiempo de selección.
- Tiempo de entrenamiento.
- Tiempo total.
- Speedup.
- Tamaño del subconjunto.
