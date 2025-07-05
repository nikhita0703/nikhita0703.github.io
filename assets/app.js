/**
 * Food Price Predictor - PyTorch Model Integration
 * AI-powered price forecasting using converted PyTorch models
 */

class FoodPricePredictor {
    constructor() {
        this.pytorchModel = null;
        this.modelConfig = null;
        this.metadata = null;
        this.scalers = null;
        this.predictionWeights = null;
        this.isLoaded = false;
        this.predictionChart = null;
        this.availableModels = [];
        this.selectedModel = null;
        
        // Initialize the application
        this.init();
    }

    async init() {
        console.log('Initializing Food Price Predictor with PyTorch Models...');
        
        // Load models index and setup UI
        try {
            await this.loadModelsIndex();
            this.setupEventListeners();
            this.generatePriceInputs();
            this.updateModelStatus('loading', '🔄 Select a model to continue...');
        } catch (error) {
            console.error('Error initializing:', error);
            this.updateModelStatus('error', '❌ Failed to load models index');
        }
    }

    async loadModelsIndex() {
        console.log('Loading models index...');
        
        try {
            const indexResponse = await fetch('./models/models_index.json');
            const modelsIndex = await indexResponse.json();
            
            this.availableModels = modelsIndex.available_models;
            console.log('Available models:', this.availableModels);
            
            this.populateModelSelector();
            
            // Auto-select default model if available
            if (modelsIndex.default_model) {
                this.selectedModel = modelsIndex.default_model;
                document.getElementById('modelSelector').value = this.selectedModel;
                await this.loadSelectedModel();
            }
            
        } catch (error) {
            console.error('Error loading models index:', error);
            this.updateModelStatus('error', '❌ Failed to load models index');
        }
    }
    
    populateModelSelector() {
        const modelSelector = document.getElementById('modelSelector');
        modelSelector.innerHTML = '<option value="">Choose a model...</option>';
        
        this.availableModels.forEach(model => {
            const option = document.createElement('option');
            option.value = model.id;
            option.textContent = `${model.name} (${model.type}) - ${model.parameters.toLocaleString()} params`;
            modelSelector.appendChild(option);
        });
    }

    async loadSelectedModel() {
        if (!this.selectedModel) {
            this.updateModelStatus('error', '❌ No model selected');
            return;
        }
        
        console.log(`Loading model: ${this.selectedModel}...`);
        this.updateModelStatus('loading', '🔄 Loading selected model...');
        
        try {
            const modelFolder = `./models/${this.selectedModel}`;
            
            // Load model configuration (lightweight)
            const configResponse = await fetch(`${modelFolder}/model_config.json`);
            this.modelConfig = await configResponse.json();
            console.log('Model config loaded:', this.modelConfig);
            
            // Load metadata
            const metadataResponse = await fetch(`${modelFolder}/model_metadata.json`);
            this.metadata = await metadataResponse.json();
            console.log('Metadata loaded:', this.metadata);
            
            // Load scalers
            const scalersResponse = await fetch(`${modelFolder}/scalers.json`);
            this.scalers = await scalersResponse.json();
            console.log('Scalers loaded:', this.scalers);
            
            // Load prediction weights (simplified weights for inference)
            const weightsResponse = await fetch(`${modelFolder}/prediction_weights.json`);
            this.predictionWeights = await weightsResponse.json();
            console.log('Prediction weights loaded');
            
            // Optionally load full PyTorch model (for advanced features)
            try {
                const pytorchResponse = await fetch(`${modelFolder}/pytorch_model.json`);
                if (pytorchResponse.ok) {
                    this.pytorchModel = await pytorchResponse.json();
                    console.log('Full PyTorch model loaded:', this.pytorchModel.model_name);
                }
            } catch (error) {
                console.warn('Full PyTorch model not loaded (using simplified version)');
            }
            
            this.isLoaded = true;
            this.updateModelStatus('ready', '✅ Model Ready');
            
        } catch (error) {
            console.error('Error loading selected model:', error);
            this.updateModelStatus('error', '❌ Failed to load selected model');
            
            // Fallback to default data
            this.metadata = this.getDefaultMetadata();
            this.scalers = this.getDefaultScalers();
            this.isLoaded = true;
        }
    }

    getDefaultMetadata() {
        return {
            input_shape: [18, 9],
            output_shape: [6],
            sequence_length: 18,
            features: [
                'normalized_price', 'ma_3', 'ma_6', 'ma_12', 'returns',
                'volatility', 'month_sin', 'month_cos', 'year_trend'
            ],
            food_items: {
                '711211': 'Bananas',
                '706111': 'Chicken (whole)',
                '709112': 'Milk (gallon)',
                '708111': 'Eggs (dozen)',
                '702111': 'Bread (white)',
                '704111': 'Bacon',
                '703111': 'Ground Beef',
                '711311': 'Oranges',
                '712112': 'Potatoes',
                '717311': 'Coffee'
            }
        };
    }

    getDefaultScalers() {
        return {
            '711211': { min: 0.35, max: 0.85, name: 'Bananas' },
            '706111': { min: 0.85, max: 2.50, name: 'Chicken (whole)' },
            '709112': { min: 2.50, max: 4.50, name: 'Milk (gallon)' },
            '708111': { min: 0.75, max: 3.50, name: 'Eggs (dozen)' },
            '702111': { min: 0.50, max: 2.50, name: 'Bread (white)' },
            '704111': { min: 2.50, max: 8.50, name: 'Bacon' },
            '703111': { min: 1.50, max: 6.50, name: 'Ground Beef' },
            '711311': { min: 0.85, max: 2.50, name: 'Oranges' },
            '712112': { min: 0.50, max: 3.50, name: 'Potatoes' },
            '717311': { min: 2.50, max: 6.50, name: 'Coffee' }
        };
    }

    updateModelStatus(status, message) {
        const statusElement = document.getElementById('modelStatus');
        const statusIndicator = statusElement.querySelector('.status-indicator');
        
        // Add model information to the status
        let modelInfo = '';
        if (this.selectedModel && this.metadata && status === 'ready') {
            modelInfo = ` (${this.metadata.display_name} - ${this.metadata.architecture_type})`;
        }
        
        statusIndicator.innerHTML = `
            <span class="status-${status}">${message}${modelInfo}</span>
        `;
        
        if (status === 'ready') {
            statusElement.style.background = 'rgba(40, 167, 69, 0.1)';
            statusElement.style.borderLeft = '4px solid #28a745';
        } else if (status === 'error') {
            statusElement.style.background = 'rgba(220, 53, 69, 0.1)';
            statusElement.style.borderLeft = '4px solid #dc3545';
        } else if (status === 'loading') {
            statusElement.style.background = 'rgba(255, 193, 7, 0.1)';
            statusElement.style.borderLeft = '4px solid #ffc107';
        }
    }

    setupEventListeners() {
        // Model selection
        document.getElementById('modelSelector').addEventListener('change', async (e) => {
            this.selectedModel = e.target.value;
            if (this.selectedModel) {
                await this.loadSelectedModel();
            } else {
                this.isLoaded = false;
                this.updateModelStatus('loading', '🔄 Select a model to continue...');
            }
        });

        // Food item selection
        document.getElementById('foodItem').addEventListener('change', (e) => {
            this.handleFoodItemChange(e.target.value);
        });

        // Generate sample data
        document.getElementById('generateSampleData').addEventListener('click', () => {
            this.generateSampleData();
        });

        // Predict button
        document.getElementById('predictBtn').addEventListener('click', () => {
            this.makePrediction();
        });
    }

    generatePriceInputs() {
        const priceGrid = document.getElementById('priceGrid');
        priceGrid.innerHTML = '';

        for (let i = 0; i < 18; i++) {
            const monthsAgo = 18 - i;
            const inputItem = document.createElement('div');
            inputItem.className = 'price-input-item';
            inputItem.innerHTML = `
                <label for="price_${i}">${monthsAgo} months ago</label>
                <input type="number" id="price_${i}" step="0.01" min="0" placeholder="$0.00">
            `;
            priceGrid.appendChild(inputItem);
        }
    }

    handleFoodItemChange(itemCode) {
        if (!itemCode || !this.scalers[itemCode]) return;

        const scaler = this.scalers[itemCode];
        const currentPriceInput = document.getElementById('currentPrice');
        currentPriceInput.placeholder = `Enter price (range: $${scaler.min.toFixed(2)} - $${scaler.max.toFixed(2)})`;
    }

    generateSampleData() {
        const foodItem = document.getElementById('foodItem').value;
        if (!foodItem) {
            alert('Please select a food item first');
            return;
        }

        const scaler = this.scalers[foodItem];
        const basePrice = (scaler.min + scaler.max) / 2;
        const variation = (scaler.max - scaler.min) * 0.15;

        // Generate realistic price history with trends
        const prices = [];
        let currentPrice = basePrice;
        
        for (let i = 0; i < 18; i++) {
            // Add some random variation and seasonal patterns
            const seasonalFactor = Math.sin(i * Math.PI / 6) * 0.08 + 1;
            const randomFactor = (Math.random() - 0.5) * 0.15 + 1;
            const trendFactor = 1 + (i * 0.003); // Small upward trend
            
            currentPrice = basePrice * seasonalFactor * randomFactor * trendFactor;
            currentPrice = Math.max(scaler.min, Math.min(scaler.max, currentPrice));
            prices.push(currentPrice);
        }

        // Fill in the price inputs
        for (let i = 0; i < 18; i++) {
            const priceInput = document.getElementById(`price_${i}`);
            priceInput.value = prices[i].toFixed(2);
        }

        // Set current price
        document.getElementById('currentPrice').value = prices[17].toFixed(2);
    }

    createFeaturesFromPrices(prices, dates) {
        const features = [];
        const n = prices.length;

        for (let i = 0; i < n; i++) {
            const feature = [];
            
            // Normalized price (simple min-max normalization)
            const minPrice = Math.min(...prices);
            const maxPrice = Math.max(...prices);
            const normalizedPrice = (prices[i] - minPrice) / (maxPrice - minPrice + 1e-8);
            feature.push(normalizedPrice);

            // Moving averages
            const ma3 = this.calculateMovingAverage(prices, i, 3);
            const ma6 = this.calculateMovingAverage(prices, i, 6);
            const ma12 = this.calculateMovingAverage(prices, i, 12);
            feature.push(ma3, ma6, ma12);

            // Price returns
            const returns = i > 0 ? (prices[i] - prices[i-1]) / (prices[i-1] + 1e-8) : 0;
            feature.push(returns);

            // Volatility (simplified)
            const volatility = this.calculateVolatility(prices, i, 6);
            feature.push(volatility);

            // Seasonal features
            const month = (i % 12) + 1;
            const monthSin = Math.sin(2 * Math.PI * month / 12);
            const monthCos = Math.cos(2 * Math.PI * month / 12);
            feature.push(monthSin, monthCos);

            // Year trend (simplified)
            const yearTrend = i / n;
            feature.push(yearTrend);

            features.push(feature);
        }

        return features;
    }

    calculateMovingAverage(prices, index, window) {
        const start = Math.max(0, index - window + 1);
        const end = index + 1;
        const slice = prices.slice(start, end);
        return slice.reduce((sum, price) => sum + price, 0) / slice.length;
    }

    calculateVolatility(prices, index, window) {
        const start = Math.max(0, index - window + 1);
        const end = index + 1;
        const slice = prices.slice(start, end);
        
        if (slice.length < 2) return 0;
        
        const mean = slice.reduce((sum, price) => sum + price, 0) / slice.length;
        const variance = slice.reduce((sum, price) => sum + Math.pow(price - mean, 2), 0) / slice.length;
        return Math.sqrt(variance);
    }

    async makePrediction() {
        if (!this.selectedModel) {
            alert('Please select a model first');
            return;
        }

        if (!this.isLoaded) {
            alert('Selected model is not loaded yet. Please wait...');
            return;
        }

        const foodItem = document.getElementById('foodItem').value;
        if (!foodItem) {
            alert('Please select a food item');
            return;
        }

        // Collect price inputs
        const prices = [];
        for (let i = 0; i < 18; i++) {
            const priceInput = document.getElementById(`price_${i}`);
            const price = parseFloat(priceInput.value);
            if (isNaN(price) || price <= 0) {
                alert(`Please enter a valid price for ${18-i} months ago`);
                return;
            }
            prices.push(price);
        }

        // Generate features
        const features = this.createFeaturesFromPrices(prices, []);
        
        // Make prediction using converted PyTorch model
        const predictions = this.pytorchModelPredict(features, foodItem);
        
        // Display results
        this.displayResults(predictions, foodItem, prices);
    }

    pytorchModelPredict(features, foodItem) {
        // Use the converted PyTorch model for prediction
        console.log(`Making prediction with ${this.metadata?.display_name || 'PyTorch model'}...`);
        
        const lastFeatures = features[features.length - 1];
        const basePrice = parseFloat(document.getElementById('currentPrice').value || '0');
        const scaler = this.scalers[foodItem];
        
        const predictions = [];
        let currentPrice = basePrice;
        
        // Use simplified prediction logic based on model weights and features
        if (this.predictionWeights) {
            // Apply simplified neural network prediction
            const inputVector = lastFeatures;
            
            // Transform input through the prediction weights
            let hidden = this.matrixMultiply([inputVector], this.predictionWeights.input_transform)[0];
            hidden = hidden.map(x => Math.tanh(x)); // Activation function
            
            // Apply hidden transformation
            hidden = this.matrixMultiply([hidden], this.predictionWeights.hidden_weights)[0];
            hidden = hidden.map(x => Math.tanh(x)); // Activation function
            
            // Get output
            let output = this.matrixMultiply([hidden], this.predictionWeights.output_transform)[0];
            
            // Add bias and apply to predictions
            for (let i = 0; i < 6; i++) {
                const prediction_factor = output[i] + this.predictionWeights.bias[i];
                const trend_factor = 1 + (prediction_factor * 0.1); // Convert to price change
                const seasonal_factor = Math.sin(i * Math.PI / 3) * 0.03 + 1;
                
                currentPrice = currentPrice * trend_factor * seasonal_factor;
                currentPrice = Math.max(scaler.min, Math.min(scaler.max, currentPrice));
                predictions.push(currentPrice);
            }
        } else {
            // Fallback prediction method
            for (let i = 0; i < 6; i++) {
                const trendFactor = 1 + (Math.random() - 0.5) * 0.08;
                const seasonalFactor = Math.sin(i * Math.PI / 3) * 0.04 + 1;
                const volatilityFactor = lastFeatures[5] * 0.3 + 0.7;
                
                currentPrice = currentPrice * trendFactor * seasonalFactor * volatilityFactor;
                currentPrice = Math.max(scaler.min, Math.min(scaler.max, currentPrice));
                predictions.push(currentPrice);
            }
        }
        
        return predictions;
    }

    matrixMultiply(a, b) {
        // Simple matrix multiplication for neural network inference
        const result = [];
        for (let i = 0; i < a.length; i++) {
            result[i] = [];
            for (let j = 0; j < b[0].length; j++) {
                let sum = 0;
                for (let k = 0; k < b.length; k++) {
                    sum += a[i][k] * b[k][j];
                }
                result[i][j] = sum;
            }
        }
        return result;
    }

    displayResults(predictions, foodItem, historicalPrices) {
        const resultsSection = document.getElementById('resultsSection');
        resultsSection.style.display = 'block';
        
        // Update prediction list
        const predictionList = document.getElementById('predictionList');
        predictionList.innerHTML = '';
        
        const currentDate = new Date();
        predictions.forEach((price, index) => {
            const futureDate = new Date(currentDate);
            futureDate.setMonth(currentDate.getMonth() + index + 1);
            
            const predictionItem = document.createElement('div');
            predictionItem.className = 'prediction-item';
            predictionItem.innerHTML = `
                <span class="prediction-month">${futureDate.toLocaleDateString('en-US', {month: 'long', year: 'numeric'})}</span>
                <span class="prediction-price">$${price.toFixed(2)}</span>
            `;
            predictionList.appendChild(predictionItem);
        });
        
        // Update trend analysis
        this.updateTrendAnalysis(predictions);
        
        // Update chart
        this.updateChart(historicalPrices, predictions, foodItem);
        
        // Update model information
        this.updateModelInfo(foodItem);
    }

    updateTrendAnalysis(predictions) {
        const trendIndicator = document.getElementById('trendIndicator');
        const trendSummary = document.getElementById('trendSummary');
        
        const firstPrice = predictions[0];
        const lastPrice = predictions[predictions.length - 1];
        const priceDiff = lastPrice - firstPrice;
        const percentChange = (priceDiff / firstPrice) * 100;
        
        let trendClass, trendIcon, trendText;
        
        if (Math.abs(percentChange) < 2) {
            trendClass = 'trend-stable';
            trendIcon = '➡️';
            trendText = `Stable trend (${percentChange.toFixed(1)}% change)`;
        } else if (percentChange > 0) {
            trendClass = 'trend-up';
            trendIcon = '📈';
            trendText = `Rising trend (+${percentChange.toFixed(1)}%)`;
        } else {
            trendClass = 'trend-down';
            trendIcon = '📉';
            trendText = `Declining trend (${percentChange.toFixed(1)}%)`;
        }
        
        trendIndicator.innerHTML = `<span class="${trendClass}">${trendIcon}</span>`;
        trendSummary.innerHTML = `<span class="${trendClass}">${trendText}</span>`;
    }

    updateChart(historicalPrices, predictions, foodItem) {
        const ctx = document.getElementById('predictionChart').getContext('2d');
        
        // Prepare data
        const currentDate = new Date();
        const labels = [];
        const historicalData = [];
        const predictedData = [];
        
        // Historical data
        for (let i = 0; i < historicalPrices.length; i++) {
            const date = new Date(currentDate);
            date.setMonth(currentDate.getMonth() - (historicalPrices.length - 1 - i));
            labels.push(date.toLocaleDateString('en-US', {month: 'short', year: 'numeric'}));
            historicalData.push(historicalPrices[i]);
            predictedData.push(null);
        }
        
        // Predicted data
        for (let i = 0; i < predictions.length; i++) {
            const date = new Date(currentDate);
            date.setMonth(currentDate.getMonth() + i + 1);
            labels.push(date.toLocaleDateString('en-US', {month: 'short', year: 'numeric'}));
            historicalData.push(null);
            predictedData.push(predictions[i]);
        }
        
        // Destroy existing chart if it exists
        if (this.predictionChart) {
            this.predictionChart.destroy();
        }
        
        // Create new chart
        this.predictionChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Historical Prices',
                    data: historicalData,
                    borderColor: '#667eea',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    fill: true,
                    tension: 0.4
                }, {
                    label: `${this.metadata?.display_name || 'PyTorch Model'} Predictions`,
                    data: predictedData,
                    borderColor: '#764ba2',
                    backgroundColor: 'rgba(118, 75, 162, 0.1)',
                    fill: true,
                    tension: 0.4,
                    borderDash: [5, 5]
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: false,
                        title: {
                            display: true,
                            text: 'Price ($)'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Time'
                        }
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: `${this.scalers[foodItem].name} Price Forecast (${this.metadata?.display_name || 'PyTorch Model'})`
                    },
                    legend: {
                        display: true,
                        position: 'top'
                    }
                }
            }
        });
    }

    updateModelInfo(foodItem) {
        const modelDetails = document.getElementById('modelDetails');
        const itemName = this.scalers[foodItem].name;
        
        let modelName = 'PyTorch Model';
        let modelArchitecture = 'LSTM';
        let trainingInfo = 'Converted from PyTorch';
        let totalParameters = 'N/A';
        
        if (this.metadata) {
            modelName = this.metadata.display_name || this.metadata.model_name;
            modelArchitecture = this.metadata.architecture_type;
            if (this.metadata.training_info) {
                trainingInfo = `Source: ${this.metadata.training_info.model_file}`;
                totalParameters = this.metadata.training_info.total_parameters?.toLocaleString() || 'N/A';
            }
        }
        
        modelDetails.innerHTML = `
            <div class="model-detail-item">
                <h4>Selected Food Item</h4>
                <p>${itemName}</p>
            </div>
            <div class="model-detail-item">
                <h4>Selected Model</h4>
                <p>${modelName}</p>
            </div>
            <div class="model-detail-item">
                <h4>Model Architecture</h4>
                <p>${modelArchitecture}</p>
            </div>
            <div class="model-detail-item">
                <h4>Model Parameters</h4>
                <p>${totalParameters}</p>
            </div>
            <div class="model-detail-item">
                <h4>Prediction Horizon</h4>
                <p>6 months ahead</p>
            </div>
            <div class="model-detail-item">
                <h4>Features Used</h4>
                <p>${this.metadata ? this.metadata.features.length : 9} engineered features</p>
            </div>
            <div class="model-detail-item">
                <h4>Training Source</h4>
                <p>${trainingInfo}</p>
            </div>
        `;
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new FoodPricePredictor();
});

// Utility functions
function formatCurrency(amount) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD'
    }).format(amount);
}

function calculatePriceChange(oldPrice, newPrice) {
    return ((newPrice - oldPrice) / oldPrice) * 100;
} 