# Semana 1 — Revisión bibliográfica y preparación del entorno

## Artículo base

Mirzasoleiman, B., Bilmes, J., & Leskovec, J. (2020).  
*Coresets for Data-efficient Training of Machine Learning Models*. ICML.

## Problema

El entrenamiento de modelos de aprendizaje automático puede ser costoso cuando el dataset es grande, ya que muchos métodos requieren calcular gradientes sobre grandes cantidades de datos.

## Idea central de CRAIG

CRAIG busca seleccionar un subconjunto ponderado de datos que aproxime el gradiente completo del dataset. En vez de entrenar con todos los puntos, se entrena con un subconjunto representativo cuyos pesos indican cuántos puntos originales representa cada elemento seleccionado.

## Objetivo matemático

Se desea aproximar:

\[
\sum_{i \in V} \nabla f_i(w)
\]

mediante:

\[
\sum_{j \in S} \gamma_j \nabla f_j(w)
\]

donde:

- \(V\) es el dataset completo.
- \(S\) es el subconjunto seleccionado.
- \(\gamma_j\) es el peso del punto seleccionado \(j\).

## Interpretación

Cada punto del dataset induce una dirección de actualización en el modelo. CRAIG selecciona puntos cuyos gradientes representan bien a los gradientes del resto del dataset.

## Implementación inicial

La primera versión usará una aproximación mediante distancias euclidianas entre puntos dentro de cada clase. Esta implementación servirá como una versión base para validar el flujo experimental antes de extender el método hacia gradientes reales o modelos neuronales.

## Baselines

1. Full data.
2. Random subset.
3. CRAIG subset.

## Datasets iniciales

1. `make_classification` de scikit-learn.
2. Breast Cancer Dataset.
3. Digits Dataset.
4. Covtype, posteriormente.

## Modelos iniciales

1. Logistic Regression.
2. SGDClassifier.
3. Red neuronal simple en PyTorch, en etapas posteriores.
