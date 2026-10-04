import pandas as pd
from pathlib import Path

SYMPTOM_COLS = ["Symptom_1", "Symptom_2", "Symptom_3"]



def load_raw_data(path: str | Path) -> pd.DataFrame:
    """Загружает CSV с датасетом."""
    return pd.read_csv(path)


def split_blood_pressure(df: pd.DataFrame) -> pd.DataFrame:
    """Разбивает '132/91' на два числовых столбца."""
    bp = df["Blood_Pressure_mmHg"].str.split("/", expand=True)
    df["systolic"] = bp[0].astype(int)
    df["diastolic"] = bp[1].astype(int)
    return df.drop(columns=["Blood_Pressure_mmHg"])


def encode_gender(df: pd.DataFrame) -> pd.DataFrame:
    """Кодирует пол: Male=0, Female=1."""
    df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})
    return df


def encode_symptoms(df: pd.DataFrame, symptom_cols: list[str]) -> pd.DataFrame:
    """One-Hot Encoding для симптомов."""
    combined = pd.concat([df[col] for col in symptom_cols])
    unique_symptoms = combined.dropna().unique()

    for symptom in unique_symptoms:
        col_name = f"symptom_{symptom.lower().replace(' ', '_')}"
        df[col_name] = (
            df[symptom_cols].eq(symptom).any(axis=1).astype(int)
        )
    return df.drop(columns=symptom_cols)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Полный пайплайн предобработки."""
    df = split_blood_pressure(df)
    df = encode_gender(df)
    df = encode_symptoms(df, SYMPTOM_COLS)
    return df


def get_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Отделяет признаки от целевой переменной."""
    drop_cols = ["Patient_ID", "Diagnosis", "Severity", "Treatment_Plan"]
    X = df.drop(columns=drop_cols)
    y = df["Diagnosis"]
    return X, y


def load_data(path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Полный пайплайн: загрузка → предобработка → разделение."""
    df = load_raw_data(path)
    df = preprocess(df)
    return get_features_and_target(df)