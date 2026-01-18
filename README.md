# A Multi-Model Approach to Carbon Footprint Prediction Using Machine Learning with Explainable AI

A machine learning application that predicts individual carbon footprints based on lifestyle factors, with explainable AI capabilities powered by SHAP values.

## Overview

This project implements a multi-model architecture comparison system that evaluates five different machine learning approaches to find the optimal model for carbon footprint prediction. The selected model is deployed via a Flask web application with real-time SHAP-based explanations for each prediction.

## Key Features

- **Multi-Model Comparison**: Automatically trains and evaluates Random Forest, XGBoost, Simple Neural Network, Deep Neural Network, and Ridge Regression models
- **Explainable AI**: SHAP (SHapley Additive exPlanations) integration provides transparency into prediction factors
- **Web Interface**: Clean, responsive Flask application for real-time predictions
- **Docker Support**: Containerized training and deployment pipelines
- **Feature Importance Analysis**: Identifies which lifestyle factors contribute most to carbon emissions

## Model Performance

The system automatically selects the best-performing model based on validation metrics. Benchmark results on the carbon footprint dataset:

| Model | R² Score | RMSE | Accuracy (±15%) |
|-------|----------|------|-----------------|
| Ridge Regression | 0.9832 | 2.13 | 98.7% |
| Simple Neural Network | 0.9805 | 2.30 | 98.7% |
| Deep Neural Network | 0.9785 | 2.42 | 98.7% |
| XGBoost | 0.9515 | 3.63 | 90.0% |
| Random Forest | 0.8771 | 5.77 | 73.3% |

## Input Features

The model considers the following lifestyle factors:

**Transportation**
- Personal Vehicle Km (annual distance driven)
- Public Vehicle Km (annual public transport usage)
- Plane Journey Count (number of flights per year)
- Train Journey Count (number of train trips per year)

**Energy and Resources**
- Electricity Kwh (monthly electricity consumption)
- Water Usage Liters (monthly water consumption)
- Waste Kg (weekly waste production)

**Lifestyle**
- Diet Type (Vegan, Vegetarian, Non-Vegetarian, etc.)

## Installation

### Prerequisites

- Python 3.10 or higher
- Docker (optional, for containerized deployment)

### Local Setup

1. Clone the repository:
```bash
git clone https://github.com/devanshishah2023-del/A-Multi-Model-Approach-to-Carbon-Footprint-Prediction-Using-Machine-Learning-with-Explainable-AI.git
cd A-Multi-Model-Approach-to-Carbon-Footprint-Prediction-Using-Machine-Learning-with-Explainable-AI/carbon_project
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Train the model:
```bash
python train_model.py
```

5. Run the web application:
```bash
python app.py
```

6. Open your browser and navigate to `http://127.0.0.1:5000`

### Docker Setup

1. Build the training container:
```bash
docker build -t carbon-xgboost .
```

2. Train the model:
```bash
docker run --rm -v $(pwd):/app/output carbon-xgboost
```

3. Run the web application:
```bash
python app.py
```

## Project Structure

```
carbon_project/
├── train_model.py        # Multi-model training pipeline
├── app.py                # Flask web application
├── requirements.txt      # Python dependencies
├── Dockerfile            # Container configuration
├── carbon_model.pkl      # Trained model (generated)
├── static/
│   ├── style.css         # Application styles
│   └── script.js         # Frontend JavaScript
└── templates/
    └── index.html        # Web interface template
```

## How It Works

### Training Pipeline

1. **Data Loading**: Fetches the carbon footprint dataset from Kaggle
2. **Preprocessing**: Handles missing values, encodes categorical features, and scales numerical inputs
3. **Model Training**: Trains five different model architectures with cross-validation
4. **Model Selection**: Automatically selects the best model based on validation R² score
5. **SHAP Analysis**: Generates feature importance rankings and stores explainability data

### Prediction Pipeline

1. User inputs lifestyle data through the web interface
2. Features are encoded and scaled using the same transformations from training
3. The model generates a carbon footprint prediction in kg CO2e
4. SHAP values are computed to explain which factors contributed most to the prediction

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main web interface |
| `/predict` | POST | Submit prediction request |
| `/feature_importance` | GET | Get global feature importance |
| `/model_info` | GET | Get model metadata and metrics |

## Dataset

This project uses the [Carbon Footprint Regression Dataset](https://www.kaggle.com/datasets/sanjaybora143/carbon-footprint-regression) from Kaggle, which contains 1000 samples of lifestyle factors and their corresponding carbon footprints.

## Contributing

Contributions are welcome. Please read the [Contributing Guidelines](CONTRIBUTING.md) before submitting a pull request.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

### Ideas for Contribution

- Add more model architectures (LightGBM, CatBoost, etc.)
- Implement hyperparameter tuning with Optuna or GridSearch
- Add unit tests for the prediction pipeline
- Create a REST API with FastAPI
- Add data visualization dashboards
- Implement user authentication for tracking personal footprints
- Deploy to cloud platforms (AWS, GCP, Azure)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- SHAP library for explainable AI capabilities
- Kaggle for hosting the carbon footprint dataset
- The scikit-learn, PyTorch, and XGBoost teams for their machine learning frameworks
