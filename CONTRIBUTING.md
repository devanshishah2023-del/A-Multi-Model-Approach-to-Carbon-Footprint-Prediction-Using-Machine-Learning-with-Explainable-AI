# Contributing to Carbon Footprint Predictor

Thank you for your interest in contributing to this project. This document provides guidelines and instructions for contributing.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Set up the development environment following the README instructions
4. Create a new branch for your changes

## Development Setup

```bash
# Clone your fork
git clone https://github.com/your-username/A-Multi-Model-Approach-to-Carbon-Footprint-Prediction-Using-Machine-Learning-with-Explainable-AI.git
cd A-Multi-Model-Approach-to-Carbon-Footprint-Prediction-Using-Machine-Learning-with-Explainable-AI

# Add upstream remote
git remote add upstream https://github.com/devanshishah2023-del/A-Multi-Model-Approach-to-Carbon-Footprint-Prediction-Using-Machine-Learning-with-Explainable-AI.git

# Navigate to project folder
cd carbon_project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Types of Contributions

### Bug Reports

Before submitting a bug report:
- Check the existing issues to avoid duplicates
- Include your Python version, OS, and relevant package versions
- Provide a minimal reproducible example if possible

### Feature Requests

Feature requests are welcome. Please provide:
- A clear description of the proposed feature
- The use case or problem it solves
- Any relevant examples or references

### Code Contributions

#### Adding New Models

To add a new model architecture to the comparison:

1. Add the model class or import in `train_model.py`
2. Include it in the model comparison loop
3. Ensure it follows the same prediction interface (sklearn-style `.predict()` or PyTorch forward pass)
4. Update the README with benchmark results

#### Improving the Web Interface

Frontend changes should:
- Maintain responsive design
- Work without JavaScript frameworks (vanilla JS only)
- Follow existing code style

#### Documentation

Documentation improvements are always welcome:
- Fix typos or unclear explanations
- Add examples or tutorials
- Translate documentation

## Code Style

### Python

- Follow PEP 8 guidelines
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Keep functions focused and reasonably sized

### JavaScript

- Use vanilla JavaScript (no frameworks)
- Prefer `const` and `let` over `var`
- Add comments for complex logic

### CSS

- Use existing class naming conventions
- Maintain mobile responsiveness
- Avoid inline styles

## Commit Messages

Write clear, concise commit messages:

```
Add XGBoost hyperparameter tuning

- Implement grid search for learning rate and max depth
- Add early stopping to prevent overfitting
- Update benchmark results in README
```

## Pull Request Process

1. Update the README if your changes affect usage or installation
2. Add or update tests if applicable
3. Ensure the training pipeline runs without errors
4. Request review from maintainers

### PR Checklist

- [ ] Code follows the project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated if needed
- [ ] No new warnings or errors introduced

## Testing

Before submitting a PR, verify:

```bash
# Train the model successfully
python train_model.py

# Run the web application
python app.py
# Test manually at http://127.0.0.1:5000
```

## Suggested Contributions

Here are some ideas if you want to contribute but are not sure where to start:

- Add LightGBM or CatBoost to the model comparison
- Implement hyperparameter tuning with Optuna
- Add unit tests using pytest
- Create a FastAPI version of the web service
- Add data visualization for model comparisons
- Implement CI/CD with GitHub Actions
- Add Dockerfile for the Flask application
- Create a demo deployment on Streamlit Cloud or Hugging Face Spaces

## Questions

If you have questions about contributing, open a discussion or issue on GitHub.

## Code of Conduct

Be respectful and constructive in all interactions. Focus on the code and ideas, not individuals. Welcome newcomers and help them get started.
