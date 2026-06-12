import pandas as pd

from src.data_utils import (
    load_synthetic_classification,
    prepare_train_test_split,
)

from src.baselines import (
    train_full_logistic,
    train_random_subset_logistic,
)


def run_initial_baseline_experiment():
    """
    Primer experimento del proyecto.

    Compara:
    - Logistic Regression con todos los datos.
    - Logistic Regression con subconjuntos aleatorios.

    Todavía no incluye CRAIG.
    """

    X, y = load_synthetic_classification(
        n_samples=5000,
        n_features=30,
        n_informative=20,
        n_redundant=5,
        n_classes=2,
        random_state=42,
    )

    X_train, X_test, y_train, y_test = prepare_train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        scale=True,
    )

    results = []

    full_model, full_result = train_full_logistic(
        X_train,
        y_train,
        X_test,
        y_test,
        dataset_name="synthetic",
    )

    results.append(full_result.to_dict())

    full_time = full_result.total_time

    fractions = [0.05, 0.10, 0.20, 0.30, 0.50]

    for fraction in fractions:
        random_model, random_result, random_indices = train_random_subset_logistic(
            X_train,
            y_train,
            X_test,
            y_test,
            fraction=fraction,
            dataset_name="synthetic",
            full_time=full_time,
        )

        results.append(random_result.to_dict())

    df = pd.DataFrame(results)

    output_path = "results/tables/initial_baselines.csv"
    df.to_csv(output_path, index=False)

    print(df)
    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    run_initial_baseline_experiment()