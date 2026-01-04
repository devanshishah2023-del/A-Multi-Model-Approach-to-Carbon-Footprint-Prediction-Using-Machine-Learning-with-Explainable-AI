/*
JavaScript for Carbon Footprint Predictor with Explainable AI
Handles form submission, API calls, result display, and SHAP explanations
*/

// Wait for DOM to be fully loaded before running code
document.addEventListener('DOMContentLoaded', function() {
    
    // Get form element
    const form = document.getElementById('predictionForm');
    
    // Get result elements
    const resultsSection = document.getElementById('resultsSection');
    const predictionValue = document.getElementById('predictionValue');
    const levelIndicator = document.getElementById('levelIndicator');
    const description = document.getElementById('description');
    const loadingOverlay = document.getElementById('loadingOverlay');
    const explanationCard = document.getElementById('explanationCard');
    const explanationList = document.getElementById('explanationList');
    
    // Add event listener for form submission
    form.addEventListener('submit', function(event) {
        // Prevent default form submission behavior
        event.preventDefault();
        
        // Call function to make prediction
        makePrediction();
    });
    
    // Function to collect form data and make prediction
    function makePrediction() {
        // Show loading overlay
        loadingOverlay.classList.add('active');
        
        // Create FormData object from form
        const formData = new FormData(form);
        
        // Convert FormData to JSON object
        const data = {};
        formData.forEach(function(value, key) {
            data[key] = value;
        });
        
        // Make POST request to Flask backend
        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        })
        .then(function(response) {
            // Parse JSON response
            return response.json();
        })
        .then(function(result) {
            // Hide loading overlay
            loadingOverlay.classList.remove('active');
            
            // Check if prediction was successful
            if (result.success) {
                // Display results
                displayResults(result);
            } else {
                // Show error message
                alert('Error: ' + result.error);
            }
        })
        .catch(function(error) {
            // Hide loading overlay
            loadingOverlay.classList.remove('active');
            
            // Show error message
            console.error('Error:', error);
            alert('An error occurred while making the prediction. Please try again.');
        });
    }
    
    // Function to display prediction results
    function displayResults(result) {
        // Update prediction value
        predictionValue.querySelector('.value').textContent = result.prediction;
        
        // Update level indicator
        const levelBadge = levelIndicator.querySelector('.level-badge');
        levelBadge.textContent = result.level.charAt(0).toUpperCase() + result.level.slice(1);
        
        // Remove all level classes
        levelBadge.className = 'level-badge';
        
        // Add appropriate level class
        const levelClass = 'level-' + result.level.replace(' ', '-');
        levelBadge.classList.add(levelClass);
        
        // Update description
        description.textContent = result.description;
        
        // Display explanation if available
        if (result.explanation && result.explanation.length > 0) {
            displayExplanation(result.explanation);
            explanationCard.style.display = 'block';
        } else {
            explanationCard.style.display = 'none';
        }
        
        // Show results section
        resultsSection.style.display = 'block';
        
        // Scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
    
    // Function to display SHAP explanation
    function displayExplanation(explanation) {
        // Clear previous explanations
        explanationList.innerHTML = '';
        
        // Create explanation items
        explanation.forEach(function(item, index) {
            // Create container for explanation item
            const itemDiv = document.createElement('div');
            itemDiv.className = 'explanation-item';
            
            // Determine if contribution is positive or negative
            const isPositive = item.contribution > 0;
            const contributionClass = isPositive ? 'positive' : 'negative';
            
            // Create HTML for explanation item
            itemDiv.innerHTML = `
                <div class="explanation-rank">${index + 1}</div>
                <div class="explanation-content">
                    <div class="explanation-feature">${item.feature}</div>
                    <div class="explanation-value">Your value: ${item.value.toFixed(2)}</div>
                    <div class="explanation-bar-container">
                        <div class="explanation-bar ${contributionClass}" style="width: ${Math.min(Math.abs(item.contribution) * 10, 100)}%"></div>
                    </div>
                    <div class="explanation-contribution ${contributionClass}">
                        ${isPositive ? '+' : ''}${item.contribution.toFixed(3)} kg CO2e impact
                    </div>
                </div>
            `;
            
            // Add to list
            explanationList.appendChild(itemDiv);
        });
    }
    
    // Add event listener for form reset
    form.addEventListener('reset', function() {
        // Hide results section when form is reset
        setTimeout(function() {
            resultsSection.style.display = 'none';
            explanationCard.style.display = 'none';
        }, 100);
    });
});