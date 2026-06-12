from sklearn.datasets import make_classification, load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_synthetic_classification(
    n_samples=10000,
    n_features=30,
    n_informative=20,
    n_redundant=5,
    n_classes=2,
    random_state=42,
):
    """
    Crea un dataset sintético de clasificación.

    Este dataset es útil para pruebas iniciales porque:
    - no requiere descargar datos
    - podemos controlar el tamaño
    - podemos controlar el número de features
    - podemos controlar el número de clases
    """

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_classes=n_classes,
        random_state=random_state,
    )

    return X, y


def load_breast_cancer_dataset():
    """
    Carga el dataset Breast Cancer de scikit-learn.

    Es un dataset pequeño de clasificación binaria.
    Sirve para validar que el pipeline funciona con datos reales.
    """

    data = load_breast_cancer()

    X = data.data
    y = data.target

    return X, y


def load_digits_dataset():
    """
    Carga el dataset Digits de scikit-learn.

    Es un dataset de clasificación multiclase con imágenes pequeñas de dígitos.
    Cada imagen viene aplanada como vector.
    """

    data = load_digits()

    X = data.data
    y = data.target

    return X, y


def prepare_train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    scale=True,
):
    """
    Divide los datos en entrenamiento y prueba.

    También puede escalar los datos usando StandardScaler.

    Regresa:
    - X_train
    - X_test
    - y_train
    - y_test
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test