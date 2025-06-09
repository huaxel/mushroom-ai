# Mushroom Classification Project for Course AI Essentials 2025

## Introduction

This project aims to classify mushrooms as either poisonous or edible based on their physical characteristics. The project involves exploratory data analysis, model pipelines, hyperparameter tuning, and model evaluation. A simple API and frontend are also deployed to demonstrate the model's capabilities.

## Features

* **Exploratory Data Analysis (EDA)**: Conducted EDA to clean the dataset, drop columns with excessive missing values, and rows with suspect values. Generated feature importance metrics and identified key features for classification.
* **Model Pipelines**: Created modular pipelines for preprocessing and classification using scikit-learn.
* **Hyperparameter Tuning**: Optimized model parameters using RandomizedSearchCV.
* **Model Evaluation**: Compared models based on accuracy, cross-validation scores, and feature importance. Identified Extra Trees as the best-performing model (~99.8% test accuracy).
* **Feature Selection**: Reduced feature set using SelectKBest and feature importance analysis, achieving comparable performance.
* **Ensemble Methods**: Implemented VotingClassifier for model combination.
* **API Development**: Developed a FastAPI-based API (`api/app.py`) for real-time predictions.
* **Frontend**: Created a simple frontend (`api/static/index.html`) to interact with the API.
* **Deployment**: Hosted the API and frontend using Cloudflare Workers & Pages.

## Technologies Used

* **Jupyter Notebooks**: Data analysis and model development
* **Pandas**: Data manipulation and analysis
* **Matplotlib and Seaborn**: Data visualization
* **scikit-learn, XGBoost and CatBoost**: Machine learning tasks
* **FastAPI**: API development
* **Uvicorn**: API deployment
* **Cloudflare Workers & Pages**: Hosting the API and frontend

## Installation Instructions

### Step 1: Clone the Repository

Clone the repository to your local machine.

```bash
git clone https://github.com/your-repo/mushroom-ai.git
cd mushroom-ai
```

### Step 2: Install Dependencies

Install the required dependencies by running:

```bash
pip install -r requirements.txt
```

### Step 3: Start the API

Start the API by running:

```bash
uvicorn api.app:app --reload
```

### Step 4: Access the Frontend

Access the frontend by navigating to [http://localhost:8000](http://localhost:8000).

## Live Demo

A live demo of the project is hosted on Cloudflare Workers & Pages. You can access it by clicking [here](insert link).

## Usage Instructions

### Using the API

1. Start the API using the command provided above.
2. Send a POST request to the `/predict` endpoint with the required input features in JSON format.
3. Example input:

```json
{
  "feature1": "value1",
  "feature2": "value2",
  "feature3": "value3"
}
```

4. Example output:

```json
{
  "prediction": "edible"
}
```

### Using the Frontend

1. Open the frontend in your browser.
2. Enter the required input features in the form.
3. Click the "Predict" button to see the result.

## License

To be added later.

## Acknowledgments

To be added later.

## Author

Benjumea Moreno, Juan
