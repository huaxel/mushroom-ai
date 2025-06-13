# Mushroom Classification Project for Course AI Essentials 2025

## Introduction

Correctly identifying poisonous mushrooms can have life-saving applications.

Using machine learning, this project classifies mushrooms as either poisonous or edible based on their physical characteristics — achieving very high accuracy.

**Final model accuracy**: 99.97% on test set.

Interestingly, gill-related features (gill-attachment, gill-color, gill-spacing) and stem-related features (stem-width and stem-color) were the most predictive.

The project involves data preprocessing, exploratory data analysis, model pipelines, hyperparameter tuning, feature selection, ensemble models with voting classifier and model evaluation.

A simple API and frontend are also deployed to demonstrate the model's capabilities.

## Live Demo

A live demo of the project is hosted on [mushrooms.huaxel.com](https://mushroom.huaxel.com/). 

## Features

* **Exploratory Data Analysis (EDA)**: Conducted EDA to clean the dataset, drop columns with excessive missing values, and rows with suspect values. Generated feature importance metrics and identified key features for classification.
* **Model Pipelines**: Created modular pipelines for preprocessing and classification using scikit-learn.
* **Hyperparameter Tuning**: Optimized model parameters using RandomizedSearchCV.
* **Model Evaluation**: Compared models based on accuracy, cross-validation scores, and feature importance. Identified Extra Trees, Random Forest and XGBoost as the best-performing models (>99.97% test accuracy).
* **Feature Selection**: Reduced feature set using SelectKBest and feature importance analysis, achieving comparable performance.
* **Ensemble Methods**: Implemented VotingClassifier for model combination.
* **API Development**: Developed a FastAPI-based API (`api/app.py`) for real-time predictions.
* **Frontend**: Created a simple frontend (`api/static/index.html`) to interact with the API.
* **Deployment**: Hosted the API and frontend using docker and fly.io.

## Technologies Used

* **Jupyter Notebooks**: Data analysis and model development
* **Pandas**: Data manipulation and analysis
* **SciPy stats**: Statistical tests for model evaluation
* **Matplotlib and Seaborn**: Data visualization
* **scikit-learn, XGBoost and CatBoost**: Machine learning tasks
* **Joblib**: Model persistence and export for deployment.
* **FastAPI**: API development
* **Uvicorn**: API deployment
* **HTML, CSS, JavaScript**: Simple frontend for interacting with the API.
* **Docker**: Hosting the API and frontend
* **GitHub Actions**: CI/CD pipeline for automated deployment to Fly.io.
* **Fly.io**: Cloud hosting for API and frontend.

## Project structure

```
mushroom-ai/
├── api/                    # FastAPI backend and frontend files
│   ├── app.py              # Main FastAPI application
│   ├── static/             # Frontend assets (HTML, CSS, JS)
│   ├── models/             # Trained ML models and metadata
│   ├── requirements.txt    # Python dependencies for the API
│   └── Dockerfile          # Dockerfile to build the API container
│   └── docker-compose.yml  # Docker Compose file for local development and deployment
├── logs/                   # Logs and JSON files from data processing and experiments
├── notebook/               # Jupyter notebooks for analysis and modeling
├── src/                    # Additional source files, datasets, and media
├── README.md               # Project documentation
├── LICENSE                 # Project license (MIT)
└── pyproject.toml          # Project configuration (optional)
```

## Installation Instructions

### Docker install

Just run

```bash
docker run -p "8080:8080" ghcr.io/huaxel/mushroom-api:latest
```

or

```bash
docker compose up
```

after cloning the repo

### Manual install

#### Step 1: Clone the Repository

Clone the repository to your local machine.

```bash
git clone https://github.com/huaxel/mushroom-ai.git
cd mushroom-ai/api
```

#### Step 2: Install Dependencies

Install the required dependencies by running:

```bash
pip install -r requirements.txt
```

#### Step 3: Start the API

Start the API by running:

```bash
uvicorn api.app:app --reload
```

#### Step 4: Access the Frontend

Access the frontend by navigating to [http://localhost:8000](http://localhost:8000).

(if running locally via Uvicorn).

For Docker, access  [http://localhost:8080](http://localhost:8080).

## Usage Instructions

### Using the API

1. Start the API using the command provided above.
2. Send a POST request to the `/predict` endpoint with the required input features in JSON format.
3. Example input:

```json
{
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
    "season": "s"
  }
```

4. Example output:

The model returns 1 for poisonous, 0 for edible mushrooms.

```json
{
  "prediction": "1"
}
```

### Using the Frontend

1. Open the frontend in your browser.
2. Enter the required input features in the form.
3. Click the "Predict" button to see the result.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

This project was developed as part of the *AI Essentials 2025* course at Erasmus HB, taught by ir. Domien Hennion.

This repository is public to contribute to learning.

## Author

Benjumea Moreno, Juan
