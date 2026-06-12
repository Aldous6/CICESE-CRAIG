import numpy as np
from sklearn.metrics import pairwise_distances
from tqdm import tqdm


class CRAIGSelector:
    """
    Implementación inicial de CRAIG para clasificación supervisada.

    Esta versión:
    - selecciona subconjuntos por clase
    - usa distancia euclidiana como proxy de distancia entre gradientes
    - usa greedy facility-location
    - calcula pesos gamma según asignación al centro más cercano

    Nota:
    Esta es una versión inicial para validar el flujo experimental.
    No es todavía la versión optimizada ni basada en gradientes reales.
    """

    def __init__(
        self,
        fraction=0.1,
        metric="euclidean",
        random_state=42,
        verbose=True,
    ):
        self.fraction = fraction
        self.metric = metric
        self.random_state = random_state
        self.verbose = verbose

        self.selected_indices_ = None
        self.sample_weights_ = None
        self.class_results_ = {}

    def fit(self, X, y):
        """
        Selecciona el subconjunto CRAIG.

        Parámetros:
        X:
            Datos de entrenamiento.

        y:
            Etiquetas de entrenamiento.

        Guarda internamente:
        - selected_indices_
        - sample_weights_
        - class_results_
        """

        selected_global = []
        selected_weights = []

        classes = np.unique(y)

        if self.verbose:
            class_iterator = tqdm(classes, desc="CRAIG class-wise selection")
        else:
            class_iterator = classes

        for class_label in class_iterator:
            class_indices = np.where(y == class_label)[0]
            X_class = X[class_indices]

            budget = max(1, int(len(class_indices) * self.fraction))

            selected_local, gamma, assignments = self._greedy_select_class(
                X_class,
                budget,
            )

            selected_global_class = class_indices[selected_local]

            selected_global.extend(selected_global_class.tolist())
            selected_weights.extend(gamma.tolist())

            self.class_results_[class_label] = {
                "class_size": len(class_indices),
                "budget": budget,
                "selected_local": selected_local,
                "selected_global": selected_global_class,
                "gamma": gamma,
                "assignments": assignments,
            }

        self.selected_indices_ = np.array(selected_global, dtype=int)
        self.sample_weights_ = np.array(selected_weights, dtype=float)

        return self

    def transform(self, X, y):
        """
        Regresa el subconjunto seleccionado y sus pesos.

        Debe llamarse después de fit().
        """

        if self.selected_indices_ is None:
            raise RuntimeError("Primero debes ejecutar fit().")

        X_subset = X[self.selected_indices_]
        y_subset = y[self.selected_indices_]
        sample_weights = self.sample_weights_

        return X_subset, y_subset, sample_weights, self.selected_indices_

    def fit_transform(self, X, y):
        """
        Ejecuta fit() y luego transform().
        """

        self.fit(X, y)
        return self.transform(X, y)

    def _greedy_select_class(self, X_class, budget):
        """
        Selecciona representantes para una sola clase.

        Esta función implementa la parte greedy tipo facility-location.

        Parámetros:
        X_class:
            Datos pertenecientes a una sola clase.

        budget:
            Número de puntos que se seleccionarán de esa clase.

        Regresa:
        - selected_indices: índices locales seleccionados
        - gamma: pesos de cada seleccionado
        - assignments: asignación de cada punto al centro más cercano
        """

        n = X_class.shape[0]

        if budget >= n:
            selected_indices = np.arange(n)
            gamma = np.ones(n, dtype=float)
            assignments = np.arange(n)
            return selected_indices, gamma, assignments

        D = pairwise_distances(
            X_class,
            X_class,
            metric=self.metric,
        )

        selected = []
        selected_mask = np.zeros(n, dtype=bool)

        # Primer centro:
        # elegimos el medoid, o sea, el punto con menor distancia total al resto.
        first_center = np.argmin(D.sum(axis=0))

        selected.append(first_center)
        selected_mask[first_center] = True

        current_min_dist = D[:, first_center].copy()

        # Ya seleccionamos uno, faltan budget - 1.
        for _ in range(1, budget):
            best_gain = -np.inf
            best_idx = None

            for candidate in range(n):
                if selected_mask[candidate]:
                    continue

                new_min_dist = np.minimum(
                    current_min_dist,
                    D[:, candidate],
                )

                gain = current_min_dist.sum() - new_min_dist.sum()

                if gain > best_gain:
                    best_gain = gain
                    best_idx = candidate

            selected.append(best_idx)
            selected_mask[best_idx] = True

            current_min_dist = np.minimum(
                current_min_dist,
                D[:, best_idx],
            )

        selected_indices = np.array(selected, dtype=int)

        D_to_selected = D[:, selected_indices]
        assignments = np.argmin(D_to_selected, axis=1)

        gamma = np.bincount(
            assignments,
            minlength=len(selected_indices),
        ).astype(float)

        return selected_indices, gamma, assignments