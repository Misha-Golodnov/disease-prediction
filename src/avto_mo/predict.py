import joblib
import pandas as pd
from pathlib import Path
from src.avto_mo.data_loader import SYMPTOM_COLS


def main() -> None:
    model_path = Path("models/disease_model.pkl")
    bundle = joblib.load(model_path)
    model = bundle["model"]
    scaler = bundle["scaler"]
    features = bundle["features"]

    print("Enter patient data:")
    age = int(input("  Age: "))
    gender_raw = input("  Gender (M/F): ").strip().upper()
    gender = 0 if gender_raw == "M" else 1
    heart_rate = int(input("  Heart rate (bpm): "))
    temperature = float(input("  Body temperature (°C): "))
    systolic = int(input("  Systolic BP: "))
    diastolic = int(input("  Diastolic BP: "))
    oxygen = int(input("  Oxygen saturation (%): "))
    symptoms_raw = input("  Symptoms (comma-separated): ")
    symptoms = [s.strip() for s in symptoms_raw.split(",") if s.strip()]

    # Собираем строку с теми же признаками, что были при обучении
    row = {feat: 0 for feat in features}
    row["Age"] = age
    row["Gender"] = gender
    row["Heart_Rate_bpm"] = heart_rate
    row["Body_Temperature_C"] = temperature
    row["systolic"] = systolic
    row["diastolic"] = diastolic
    row["Oxygen_Saturation_%"] = oxygen

    for symptom in symptoms:
        col = f"symptom_{symptom.lower().replace(' ', '_')}"
        if col in row:
            row[col] = 1

    X = pd.DataFrame([row])[features]
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0]
    print(f"\nPredicted diagnosis: {prediction}")


if __name__ == "__main__":
    main()