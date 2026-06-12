import time
from dataclasses import dataclass
from typing import Dict, Any, Optional

import numpy as np
from sklearn.metrics import accuracy_score, log_loss


@dataclass
class ExperimentResult:
    """
    Estructura para guardar los resultados de un experimento.

    Cada vez que entrenemos un modelo, vamos a guardar:
    - dataset usado
    - modelo usado
    - método usado: full, random o craig
    - fracción de datos usada
    - tamaño del subconjunto
    - tiempos
    - accuracy
    - log loss
    - speedup
    """

    dataset: str
    model: str
    method: str
    fraction: float
    subset_size: int
    selection_time: float
    training_time: float
    total_time: float
    accuracy: float
    log_loss: float
    speedup: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convierte el resultado a diccionario.

        Esto sirve para poder meter muchos resultados en un DataFrame de pandas.
        """
        return {
            "dataset": self.dataset,
            "model": self.model,
            "method": self.method,
            "fraction": self.fraction,
            "subset_size": self.subset_size,
            "selection_time": self.selection_time,
            "training_time": self.training_time,
            "total_time": self.total_time,
            "accuracy": self.accuracy,
            "log_loss": self.log_loss,
            "speedup": self.speedup,
        }


def evaluate_classifier(model, X_test, y_test):
    """
    Evalúa un modelo de clasificación.

    Regresa:
    - accuracy
    - log loss

    Parámetros:
    model:
        Modelo ya entrenado.

    X_test:
        Datos de prueba.

    y_test:
        Etiquetas reales de prueba.
    """

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)
        loss = log_loss(y_test, y_proba)
    else:
        loss = np.nan

    return accuracy, loss


def compute_speedup(full_time: float, method_time: float) -> float:
    """
    Calcula qué tan rápido fue un método comparado con full data.

    speedup = tiempo_full_data / tiempo_metodo

    Ejemplo:
    full_time = 10 segundos
    method_time = 2 segundos

    speedup = 10 / 2 = 5

    Eso significa que el método fue 5 veces más rápido.
    """

    if method_time <= 0:
        return np.nan

    return full_time / method_time