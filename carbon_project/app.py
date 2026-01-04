"""
Flask Web Application for Carbon Footprint Prediction
Works with ANY model type (Random Forest, XGBoost, Neural Network, etc.)
"""

from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
import numpy as np
import pickle
import os

app = Flask(__name__)

# Global variables
model = None
model_type = None
scaler = None
encoders = None
feature_names = None
num_cols = None
cat_cols = None
model_metrics = None
shap_data = None

def load_model():
    """Load the trained model and metadata from file"""
    global model, model_type, scaler, encoders, feature_names, num_cols, cat_cols, model_metrics, shap_data
    
    if not os.path.exists('carbon_model.pkl'):
        raise FileNotFoundError(
            "Model file not found. Please run train_multimodel.py first to train the model."
        )
    
    with open('carbon_model.pkl', 'rb') as f:
        model_package = pickle.load(f)
    
    model = model_package['model']
    model_type = model_package.get('model_type', 'Unknown')
    scaler = model_package['scaler']
    encoders = model_package['encoders']
    feature_names = model_package['feature_names']
    num_cols = model_package['num_cols']
    cat_cols = model_package['cat_cols']
    model_metrics = model_package['metrics']
    shap_data = model_package.get('shap_data', None)
    
    # Set PyTorch models to eval mode
    if hasattr(model, 'eval'):
        model.eval()
    
    print("Model loaded successfully!")
    print(f"Model type: {model_type}")
    print(f"Model metrics - R²: {model_metrics['r2']:.3f}, Accuracy: {model_metrics['accuracy']:.1f}%")
    
    if shap_data:
        print("SHAP explanations loaded - Explainable AI enabled!")

def generate_description(prediction):
    """Generate natural language description of prediction"""
    if prediction < 5:
        level = "very low"
    elif prediction < 10:
        level = "low"
    elif prediction < 15:
        level = "moderate"
    elif prediction < 20:
        level = "high"
    else:
        level = "very high"
    
    description = f"Your predicted carbon footprint is {prediction:.2f} kg CO2e. "
    description += f"This is considered a {level} carbon footprint."
    
    return description, level

def make_prediction(feature_values_scaled):
    """Make prediction using loaded model (works with any model type)"""
    if model_type in ['Random Forest', 'XGBoost', 'Ridge']:
        # Scikit-learn style models
        prediction = model.predict(feature_values_scaled.reshape(1, -1))[0]
    else:
        # PyTorch neural network models
        with torch.no_grad():
            feature_tensor = torch.FloatTensor(feature_values_scaled.reshape(1, -1))
            prediction = model(feature_tensor).item()
    
    return prediction

def explain_prediction(feature_values_scaled, feature_values_original):
    """Generate explanation for prediction using SHAP values"""
    if not shap_data:
        return None
    
    try:
        import shap
        
        # Create prediction function based on model type
        if model_type in ['Random Forest', 'XGBoost', 'Ridge']:
            explainer = shap.Explainer(model, shap_data['X_shap_background'])
            shap_vals = explainer(feature_values_scaled.reshape(1, -1))
            if len(shap_vals.shape) > 2:
                shap_values = shap_vals.values[0, :, 0]
            else:
                shap_values = shap_vals.values[0]
        else:
            # PyTorch model
            def predict_fn(X):
                model.eval()
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X)
                    predictions = model(X_tensor).cpu().numpy().flatten()
                return predictions
            
            explainer = shap.KernelExplainer(predict_fn, shap_data['X_shap_background'])
            shap_values = explainer.shap_values(feature_values_scaled.reshape(1, -1), nsamples=50)[0]
        
        # Get contribution of each feature
        contributions = []
        for i, feature in enumerate(feature_names):
            contributions.append({
                'feature': feature.replace('_', ' ').title(),
                'value': float(feature_values_original[i]),
                'contribution': float(shap_values[i]),
                'importance': abs(float(shap_values[i]))
            })
        
        contributions.sort(key=lambda x: x['importance'], reverse=True)
        return contributions[:5]
    
    except Exception as e:
        print(f"Error generating explanation: {e}")
        return None

@app.route('/')
def home():
    """Render the home page with input form"""
    return render_template('index.html', 
                         num_cols=num_cols, 
                         cat_cols=cat_cols,
                         encoders=encoders,
                         has_shap=shap_data is not None)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests from the form"""
    try:
        data = request.json
        
        # Prepare feature arrays
        features = {}
        features_original = {}
        
        # Process numerical features
        for col in num_cols:
            value = float(data.get(col, 0))
            features[col] = value
            features_original[col] = value
        
        # Process categorical features
        for col in cat_cols:
            value = data.get(col, '')
            if value in encoders[col].classes_:
                encoded_value = encoders[col].transform([value])[0]
            else:
                encoded_value = 0
            features[f'{col}_enc'] = encoded_value
            features_original[f'{col}_enc'] = encoded_value
        
        # Create feature arrays
        feature_array = np.array([[features[name] for name in feature_names]])
        feature_array_original = np.array([[features_original[name] for name in feature_names]])
        
        # Scale features
        feature_scaled = scaler.transform(feature_array)
        
        # Make prediction (works with any model type)
        prediction = make_prediction(feature_scaled[0])
        
        # Generate description
        description, level = generate_description(prediction)
        
        # Generate explanation if available
        explanation = None
        if shap_data:
            explanation = explain_prediction(feature_scaled[0], feature_array_original[0])
        
        return jsonify({
            'success': True,
            'prediction': round(prediction, 2),
            'description': description,
            'level': level,
            'explanation': explanation
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/feature_importance')
def feature_importance():
    """Return global feature importance from SHAP"""
    if not shap_data:
        return jsonify({
            'success': False,
            'error': 'SHAP data not available.'
        })
    
    top_features = []
    for i in range(len(shap_data['top_features'])):
        top_features.append({
            'feature': shap_data['top_features'][i].replace('_', ' ').title(),
            'importance': float(shap_data['top_importance'][i])
        })
    
    return jsonify({
        'success': True,
        'features': top_features
    })

@app.route('/model_info')
def model_info():
    """Return model information and metrics"""
    return jsonify({
        'model_type': model_type,
        'metrics': model_metrics,
        'features': {
            'numerical': num_cols,
            'categorical': cat_cols
        },
        'explainable_ai': shap_data is not None
    })

if __name__ == '__main__':
    print("Loading model...")
    load_model()
    
    print("\nStarting Flask server...")
    print("Access the application at: http://127.0.0.1:5000")
    print("Press Ctrl+C to stop the server")
    
    app.run(debug=True, host='127.0.0.1', port=5000)