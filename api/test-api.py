import joblib
import pandas as pd

print("Starting test script")

def test_model_prediction(model_path: str):
    print(f"Loading model from {model_path}...")
    # Load pipeline model
    pipeline = joblib.load("./models/XGBoost.pkl")
    print("Loaded pipeline steps:")
    for name, step in pipeline.named_steps.items():
        print(f"- {name}: {type(step)}")

    # Example input data matching your API input schema
    sample_input = pd.DataFrame([{
        "cap-diameter": 5.0,
        "cap-shape": "b",
        "cap-surface": "s",
        "cap-color": "n",
        "does-bruise-or-bleed": "f",
        "gill-attachment": "a",
        "gill-spacing": "c",
        "gill-color": "n",
        "stem-height": 7.0,
        "stem-width": 1.2,
        "stem-color": "n",
        "has-ring": "t",
        "ring-type": "c",
        "habitat": "g",
        "season": "s",
    }])

    # Predict
    prediction = pipeline.predict(sample_input)
    print("Prediction:", prediction)

if __name__ == "__main__":
    test_model_prediction("models/XGBoost.pkl")