import joblib
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

from src.avto_mo.data_loader import (
    load_raw_data,
    preprocess,
    get_features_and_target,
)


def main() -> None:
    # --- Пути ---
    data_path = Path("data/raw/disease_diagnosis.csv")
    processed_path = Path("data/processed/disease_diagnosis_processed.csv")
    model_path = Path("models/disease_model.pkl")

    # Создаём папки, если их нет
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Загрузка и предобработка ---
    df = load_raw_data(data_path)
    df = preprocess(df)

    # Сохраняем обработанный датасет
    df.to_csv(processed_path, index=False)
    print(f"Processed data saved to {processed_path}")

    # Отделяем признаки от цели
    X, y = get_features_and_target(df)
    print(f"Features: {list(X.columns)}")
    print(f"Target classes: {sorted(y.unique())}")

    # --- Разбиение на train/test ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Масштабирование ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Обучение ---
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    # --- Оценка ---
    preds = model.predict(X_test_scaled)
    print(f"\nAccuracy: {accuracy_score(y_test, preds):.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, preds, zero_division=0))

    # --- Сохранение модели ---
    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "features": list(X.columns),
        },
        model_path,
    )
    print(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    main()