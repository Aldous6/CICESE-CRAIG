import time
import numpy as np

from sklearn.linear_model import LogisticRegression

from src.metrics import evaluate_classifier, ExperimentResult


def train_full_logistic(
    X_train,
    y_train,
    X_test,
    y_test,
    dataset_name="synthetic",
    max_iter=1000,
    random_state=42,
):
    """
    Entrena regresión logística usando todos los datos de entrenamiento.

    Este método es nuestro baseline principal:
    representa el caso estándar donde no reducimos datos.
    """

    start = time.time()

    model = LogisticRegression(
        max_iter=max_iter,
        random_state=random_state,
        solver="lbfgs",
    )

    model.fit(X_train, y_train)

    training_time = time.time() - start

    accuracy, loss = evaluate_classifier(model, X_test, y_test)

    result = ExperimentResult(
        dataset=dataset_name,
        model="LogisticRegression",
        method="full",
        fraction=1.0,
        subset_size=len(X_train),
        selection_time=0.0,
        training_time=training_time,
        total_time=training_time,
        accuracy=accuracy,
        log_loss=loss,
        speedup=1.0,
    )

    return model, result


def train_random_subset_logistic(
    X_train,
    y_train,
    X_test,
    y_test,
    fraction=0.1,
    dataset_name="synthetic",
    max_iter=1000,
    random_state=42,
    full_time=None,
):
    """
    Entrena regresión logística usando un subconjunto aleatorio.

    Este método sirve para comparar CRAIG contra una selección ingenua.
    """

    rng = np.random.default_rng(random_state)

    n_train = len(X_train)
    subset_size = max(1, int(n_train * fraction))

    start_selection = time.time()

    subset_indices = rng.choice(
        n_train,
        size=subset_size,
        replace=False,
    )

    selection_time = time.time() - start_selection

    start_training = time.time()

    model = LogisticRegression(
        max_iter=max_iter,
        random_state=random_state,
        solver="lbfgs",
    )

    model.fit(
        X_train[subset_indices],
        y_train[subset_indices],
    )

    training_time = time.time() - start_training
    total_time = selection_time + training_time

    accuracy, loss = evaluate_classifier(model, X_test, y_test)

    speedup = None
    if full_time is not None:
        speedup = full_time / total_time

    result = ExperimentResult(
        dataset=dataset_name,
        model="LogisticRegression",
        method="random",
        fraction=fraction,
        subset_size=subset_size,
        selection_time=selection_time,
        training_time=training_time,
        total_time=total_time,
        accuracy=accuracy,
        log_loss=loss,
        speedup=speedup,
    )

    return model, result, subset_indices