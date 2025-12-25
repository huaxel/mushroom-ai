# Mushroom Classification Project for Course AI Essentials 2025

## Introduction

Correctly identifying poisonous mushrooms can have life-saving applications.

Using machine learning, this project classifies mushrooms as either poisonous or edible based on their physical characteristics — achieving **very high accuracy**.

 > **Final model accuracy**: 99.97% on test set.

Also interesting to note is that gill-related features (gill-attachment, gill-color, gill-spacing) and stem-related features (stem-width and stem-color) were the most predictive.

The project involves data preprocessing, exploratory data analysis, model pipelines, hyperparameter tuning, feature selection, ensemble models with voting classifier and model evaluation.

A simple API and frontend are also deployed to demonstrate the model's capabilities.

## Live Demo

A live demo of the project is hosted on [mushroom.huaxel.com](https://mushroom.huaxel.com/).

## Technical Approach

A structured approach was followed to ensure model robustness and reproducibility:

### 1. Exploratory Data Analysis (EDA) & Preprocessing

- Conducted EDA to understand feature distributions and class balance.
- Checked for missing values and analyzed categorical frequency.
- Built separate **preprocessing pipelines** for numerical and categorical features to ensure consistency between training and validation.

### 2. Model Training & Tuning

- Experimented with varying models: **Logistic Regression (with PCA), SGD, Random Forest, Extra Trees, XGBoost, and CatBoost**.
- Used **RandomizedSearchCV** for hyperparameter tuning.
- Plotted **learning curves** to verify generalization and check for over/underfitting.

### 3. Evaluation & Comparisons

- Tree-based models (Extra Trees, Random Forest, XGBoost) performed best (>99.8% test accuracy).
- Linear models underperformed, suggesting non-linear feature relationships.
- **XGBoost** was selected as the final production model for its balance of accuracy and efficiency.

### 4. Feature Optimization

- Feature importance analysis revealed a small subset of features drove most predictions.
- **SelectKBest** showed that using only the **top 6 features** yielded comparable performance to using all 15.
- While ensemble methods (VotingClassifier) were tested, the gain was marginal compared to a well-tuned XGBoost model.

## Features

- **API Development**: developed a FastAPI-based API for real-time predictions.
- **Frontend**: created a simple frontend to interact with the API.
- **Deployment**: hosted the API and frontend using docker and fly.io.
- **CI/CD**: Automated testing pipeline via GitHub Actions to ensure code quality.

## Technologies Used

- **Jupyter Notebooks**: Data analysis and model development
- **Pandas**: Data manipulation and analysis
- **SciPy stats**: Statistical tests for model evaluation
- **Matplotlib and Seaborn**: Data visualization
- **scikit-learn, XGBoost and CatBoost**: Machine learning tasks
- **Joblib**: Model persistence and export for deployment.
- **FastAPI**: API development
- **Uvicorn**: API deployment
- **HTML, CSS, JavaScript**: Simple frontend for interacting with the API.
- **Docker**: Hosting the API and frontend
- **GitHub Actions**: CI/CD pipeline for automated testing and deployment.
- **Fly.io**: Cloud hosting for API and frontend.

## Project structure

```
mushroom-ai/
├── api/                    # FastAPI backend and frontend files
│   ├── models/             # Trained ML models and metadata
│   ├── static/             # Frontend assets (HTML, CSS, JS)
│   ├── tests/              # Unit tests
│   ├── app.py              # Main FastAPI application
│   ├── schemas.py          # Pydantic models
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
cd mushroom-ai
```

#### Step 2: Install Dependencies

It is recommended to use a virtual environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r api/requirements.txt
```

#### Step 3: Run Tests

Ensure everything is working correctly by running the tests.

```bash
pip install pytest httpx
python -m pytest api/tests
```

#### Step 4: Start the API

Start the API by running:

```bash
cd api
uvicorn app:app --reload
```

#### Step 5: Access the Frontend

Access the frontend by navigating to [http://localhost:8000](http://localhost:8000).

## API Usage

Send a POST request to the `/predict` endpoint:

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

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

This project was developed as part of the *AI Essentials 2025* course at Erasmus HB, taught by ir. Domien Hennion.

This repository is public to contribute to learning.

## Author

Benjumea Moreno, Juan
