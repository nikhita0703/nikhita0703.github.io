/**
 * Grocery Price Predictor - PyTorch Model Integration
 * AI-powered price forecasting using converted PyTorch models
 * TODO: Add support for transformer models
 * FIXME: Chart performance issues with large datasets
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
        this.selectedFoodItems = new Set();
        this.lastSelectedItem = null;
        
        // TODO: Add caching for model responses
        // this.cache = new Map(); // might implement later
        
        // HACK: Quick fix for Safari compatibility
        this.isSafari = /^((?!chrome|android).)*safari/i.test(navigator.userAgent);
        
        // Initialize the application
        this.init();
    }

    async init() {
        console.log('Initializing Grocery Price Predictor with PyTorch Models...');
        
        // Load models index and setup UI
        try {
            await this.loadModelsIndex();
            this.setupEventListeners();
            this.updateModelStatus('loading', 'Select a model to continue...');
            
            // TODO: Add loading spinner here
            // this.showLoadingSpinner();
        } catch (error) {
            console.error('Error initializing:', error);
            this.updateModelStatus('error', 'Failed to load models index');
            
            // FIXME: Better error handling needed
            // Maybe show a retry button?
        }
    }

    async loadModelsIndex() {
        console.log('Loading models index...');
        
        try {
            const indexResponse = await fetch('./models/models_index.json');
            const modelsIndex = await indexResponse.json();
            
            this.availableModels = modelsIndex.available_models;
            console.log('Available models:', this.availableModels);
            
            // Old way - kept for reference
            // this.availableModels = [
            //     { id: 'lstm', name: 'LSTM Model', type: 'RNN' },
            //     { id: 'transformer', name: 'Transformer', type: 'Attention' }
            // ];
            
            this.populateModelSelector();
            
            // Auto-select default model if available
            if (modelsIndex.default_model) {
                this.selectedModel = modelsIndex.default_model;
                document.getElementById('modelSelector').value = this.selectedModel;
                await this.loadSelectedModel();
            }
            
        } catch (error) {
            console.error('Error loading models index:', error);
            this.updateModelStatus('error', 'Failed to load models index');
            
            // Quick fallback - not ideal but works
            this.availableModels = [{ id: 'fallback', name: 'Fallback Model', type: 'Basic' }];
        }
    }
    
    populateModelSelector() {
        const modelSelector = document.getElementById('modelSelector');
        modelSelector.innerHTML = '<option value="">Choose a model...</option>';
        
        this.availableModels.forEach(model => {
            const option = document.createElement('option');
            option.value = model.id;
            // FIXME: parameters might be undefined for some models
            const params = model.parameters ? model.parameters.toLocaleString() : 'Unknown';
            option.textContent = `${model.name} (${model.type}) - ${params} params`;
            modelSelector.appendChild(option);
        });
        
        // Legacy support for old jQuery users
        // if (window.jQuery) {
        //     $(modelSelector).trigger('change');
        // }
    }

    async loadSelectedModel() {
        if (!this.selectedModel) {
            this.updateModelStatus('error', 'No model selected');
            return;
        }
        
        console.log(`Loading model: ${this.selectedModel}...`);
        this.updateModelStatus('loading', 'Loading selected model...');
        
        try {
            const modelFolder = `./models/${this.selectedModel}`;
            
            const configResponse = await fetch(`${modelFolder}/model_config.json`);
            this.modelConfig = await configResponse.json();
            console.log('Model config loaded:', this.modelConfig);
            
            const metadataResponse = await fetch(`${modelFolder}/model_metadata.json`);
            this.metadata = await metadataResponse.json();
            console.log('Metadata loaded:', this.metadata);
            
            const scalersResponse = await fetch(`${modelFolder}/scalers.json`);
            this.scalers = await scalersResponse.json();
            console.log('Scalers loaded:', this.scalers);
            
            this.populateFoodItems();
            
            const weightsResponse = await fetch(`${modelFolder}/prediction_weights.json`);
            this.predictionWeights = await weightsResponse.json();
            console.log('Prediction weights loaded');
            try {
                const pytorchResponse = await fetch(`${modelFolder}/pytorch_model.json`);
                if (pytorchResponse.ok) {
                    this.pytorchModel = await pytorchResponse.json();
                    console.log('Full PyTorch model loaded:', this.pytorchModel.model_name);
                }
            } catch (error) {
                console.warn('Full PyTorch model not loaded (using simplified version)');
                // TODO: Show warning to user about limited functionality
            }
            
            // Quick hack for performance - preload Chart.js if not already loaded
            if (!window.Chart) {
                // Lazy load Chart.js
                const script = document.createElement('script');
                script.src = 'https://cdn.jsdelivr.net/npm/chart.js';
                document.head.appendChild(script);
            }
            
                        this.isLoaded = true;
            this.updateModelStatus('ready', 'Model Ready');
            
            this.updatePredictButtonState();
            
        } catch (error) {
            console.error('Error loading selected model:', error);
            this.updateModelStatus('error', 'Failed to load selected model');
            
            this.populateFoodItems();
            this.isLoaded = true;
            
            // Show user-friendly error message
            // alert('Model loading failed, using fallback data');
        }
    }
    
    populateFoodItems() {
        const foodSelector = document.getElementById('foodItem');
        foodSelector.innerHTML = '<option value="">Choose a food item...</option>';
        
        const sortedItems = Object.entries(this.scalers).sort((a, b) => 
            a[1].name.localeCompare(b[1].name)
        );
        
        sortedItems.forEach(([code, item]) => {
            const option = document.createElement('option');
            option.value = code;
            option.textContent = `${item.name} (${item.data_points} data points)`;
            foodSelector.appendChild(option);
        });
        
        console.log(`Populated ${sortedItems.length} food items`);
    }

    updateModelStatus(status, message) {
        const statusElement = document.getElementById('modelStatus');
        const statusIndicator = statusElement.querySelector('.status-indicator');
        
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
        document.getElementById('modelSelector').addEventListener('change', async (e) => {
            this.selectedModel = e.target.value;
            if (this.selectedModel) {
                await this.loadSelectedModel();
            } else {
                this.isLoaded = false;
                this.updateModelStatus('loading', 'Select a model to continue...');
            }
        });

        document.getElementById('foodItem').addEventListener('change', (e) => {
            this.handleMultipleFoodItemChange(e);
        });

        document.getElementById('clearSelectionBtn').addEventListener('click', () => {
            this.clearAllSelections();
        });

        document.getElementById('predictBtn').addEventListener('click', () => {
            this.makePrediction();
        });
        
        // HACK: Fix for Safari's weird select behavior
        if (this.isSafari) {
            document.getElementById('foodItem').addEventListener('focus', function() {
                this.size = 5; // temporary workaround
            });
            document.getElementById('foodItem').addEventListener('blur', function() {
                this.size = 1;
            });
        }
        
        // Add keyboard shortcuts - might be useful
        // document.addEventListener('keydown', (e) => {
        //     if (e.ctrlKey && e.key === 'p') {
        //         e.preventDefault();
        //         this.makePrediction();
        //     }
        // });
    }

    displayHistoricalData(itemCode) {
        const container = document.getElementById('historicalDataContainer');
        const currentPriceDisplay = document.getElementById('currentPriceDisplay');
        const predictBtn = document.getElementById('predictBtn');
        
        if (!itemCode || !this.scalers[itemCode]) {
            container.innerHTML = '<p class="data-notice">Select a food item to view historical prices</p>';
            currentPriceDisplay.style.display = 'none';
            predictBtn.disabled = true;
            return;
        }
        
        // Performance optimization - cache DOM queries
        // const cachedContainer = this.cachedElements?.container || container;
        // TODO: Implement proper caching system

        const scaler = this.scalers[itemCode];
        const historicalPrices = scaler.latest_18_months || [];
        const historicalDates = scaler.latest_dates || [];
        const currentPrice = scaler.current_price || scaler.mean;

        if (historicalPrices.length === 0) {
            container.innerHTML = '<p class="error-notice">No historical data available for this item</p>';
            currentPriceDisplay.style.display = 'none';
            predictBtn.disabled = true;
            return;
        }

        // Create historical data table
        // TODO: Convert to proper table component instead of string concatenation
        let tableHTML = `
            <div class="historical-table-container">
                <table class="historical-data-table">
                    <thead>
                        <tr>
                            <th>Date</th>
                            <th>Price</th>
                            <th>Change</th>
                        </tr>
                    </thead>
                    <tbody>
        `;

        // Original way - kept for debugging
        // let tableRows = [];
        // historicalPrices.forEach((price, i) => {
        //     const date = historicalDates[i];
        //     const row = `<tr><td>${date}</td><td>$${price.toFixed(2)}</td></tr>`;
        //     tableRows.push(row);
        // });

        for (let i = 0; i < historicalPrices.length; i++) {
            const price = historicalPrices[i];
            const date = historicalDates[i];
            const prevPrice = i > 0 ? historicalPrices[i-1] : price;
            const change = i > 0 ? ((price - prevPrice) / prevPrice * 100) : 0;
            const changeClass = change > 0 ? 'positive' : change < 0 ? 'negative' : 'neutral';
            const changeSymbol = change > 0 ? '+' : '';

            tableHTML += `
                <tr>
                    <td>${date}</td>
                    <td>$${price.toFixed(2)}</td>
                    <td class="change ${changeClass}">
                        ${changeSymbol}${change.toFixed(1)}%
                    </td>
                </tr>
            `;
        }

        tableHTML += `
                    </tbody>
                </table>
            </div>
            <div class="data-source">
                <p><strong>Data Source:</strong> US Bureau of Labor Statistics (${scaler.series_id})</p>
                <p><strong>Data Range:</strong> ${scaler.start_year} - ${scaler.end_year} (${scaler.data_points} total points)</p>
            </div>
        `;

        container.innerHTML = tableHTML;

        document.getElementById('latestPriceValue').textContent = `$${currentPrice.toFixed(2)}`;
        document.getElementById('latestPriceDate').textContent = historicalDates[historicalDates.length - 1] || 'Latest';
        currentPriceDisplay.style.display = 'block';

        predictBtn.disabled = false;

        console.log(`Displayed ${historicalPrices.length} months of historical data for ${scaler.name}`);
    }

    handleMultipleFoodItemChange(event) {
        const selectedOptions = Array.from(event.target.selectedOptions);
        const newSelectedItems = selectedOptions.map(option => option.value).filter(value => value !== '');
        
        this.selectedFoodItems.clear();
        newSelectedItems.forEach(item => this.selectedFoodItems.add(item));
        
        if (newSelectedItems.length > 0) {
            this.lastSelectedItem = newSelectedItems[newSelectedItems.length - 1];
            this.displayHistoricalData(this.lastSelectedItem);
        } else {
            this.lastSelectedItem = null;
            this.displayHistoricalData(null);
        }
        
        this.updateSelectedItemsDisplay();
        
        this.updatePredictButtonState();
    }

    updateSelectedItemsDisplay() {
        const container = document.getElementById('selectedItemsContainer');
        const list = document.getElementById('selectedItemsList');
        
        if (this.selectedFoodItems.size === 0) {
            container.style.display = 'none';
            return;
        }
        
        container.style.display = 'block';
        list.innerHTML = '';
        
        const sortedItems = Array.from(this.selectedFoodItems).sort((a, b) => 
            this.scalers[a].name.localeCompare(this.scalers[b].name)
        );
        
        sortedItems.forEach(itemCode => {
            const scaler = this.scalers[itemCode];
            const currentPrice = scaler.current_price || scaler.mean;
            
            const itemDiv = document.createElement('div');
            itemDiv.className = 'selected-item';
            itemDiv.innerHTML = `
                <div class="selected-item-info">
                    <div class="selected-item-name">${scaler.name}</div>
                    <div class="selected-item-price">Current: $${currentPrice.toFixed(2)}</div>
                </div>
                <button class="remove-item-btn" onclick="foodPredictor.removeSelectedItem('${itemCode}')">×</button>
            `;
            list.appendChild(itemDiv);
        });
        
        console.log(`Updated display for ${this.selectedFoodItems.size} selected items`);
    }

    removeSelectedItem(itemCode) {
        this.selectedFoodItems.delete(itemCode);
        
        const foodSelector = document.getElementById('foodItem');
        Array.from(foodSelector.options).forEach(option => {
            if (option.value === itemCode) {
                option.selected = false;
            }
        });
        
        if (this.lastSelectedItem === itemCode) {
            const remainingItems = Array.from(this.selectedFoodItems);
            this.lastSelectedItem = remainingItems.length > 0 ? remainingItems[remainingItems.length - 1] : null;
            this.displayHistoricalData(this.lastSelectedItem);
        }
        
        this.updateSelectedItemsDisplay();
        this.updatePredictButtonState();
    }

    clearAllSelections() {
        this.selectedFoodItems.clear();
        this.lastSelectedItem = null;
        
        const foodSelector = document.getElementById('foodItem');
        Array.from(foodSelector.options).forEach(option => {
            option.selected = false;
        });
        
        this.displayHistoricalData(null);
        this.updateSelectedItemsDisplay();
        this.updatePredictButtonState();
    }

    updatePredictButtonState() {
        const predictBtn = document.getElementById('predictBtn');
        const hasValidItems = this.selectedFoodItems.size > 0 && 
                             Array.from(this.selectedFoodItems).every(itemCode => {
                                 const scaler = this.scalers[itemCode];
                                 return scaler && scaler.latest_18_months && scaler.latest_18_months.length >= 18;
                             });
        
        predictBtn.disabled = !hasValidItems || !this.isLoaded;
        
        if (this.selectedFoodItems.size === 0) {
            predictBtn.textContent = '🔮 Predict Future Prices';
        } else if (this.selectedFoodItems.size === 1) {
            predictBtn.textContent = '🔮 Predict Future Price';
        } else {
            predictBtn.textContent = `🔮 Predict Total Cost (${this.selectedFoodItems.size} items)`;
        }
    }



    createFeaturesFromPrices(prices, dates) {
        const features = [];
        const n = prices.length;

        // Pre-calculate min/max for performance - learned this the hard way
        const minPrice = Math.min(...prices);
        const maxPrice = Math.max(...prices);

        for (let i = 0; i < n; i++) {
            const feature = [];
            
            const normalizedPrice = (prices[i] - minPrice) / (maxPrice - minPrice + 1e-8);
            feature.push(normalizedPrice);

            const ma3 = this.calculateMovingAverage(prices, i, 3);
            const ma6 = this.calculateMovingAverage(prices, i, 6);
            const ma12 = this.calculateMovingAverage(prices, i, 12);
            feature.push(ma3, ma6, ma12);

            const returns = i > 0 ? (prices[i] - prices[i-1]) / (prices[i-1] + 1e-8) : 0;
            feature.push(returns);

            const volatility = this.calculateVolatility(prices, i, 6);
            feature.push(volatility);

            const month = (i % 12) + 1;
            const monthSin = Math.sin(2 * Math.PI * month / 12);
            const monthCos = Math.cos(2 * Math.PI * month / 12);
            feature.push(monthSin, monthCos);

            const yearTrend = i / n;
            feature.push(yearTrend);

            features.push(feature);
        }

        return features;
        
        // Alternative approach using functional programming - maybe later
        // return prices.map((price, i) => {
        //     const feature = [(price - minPrice) / (maxPrice - minPrice + 1e-8)];
        //     // ... rest of features
        //     return feature;
        // });
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

        if (this.selectedFoodItems.size === 0) {
            alert('Please select at least one food item');
            return;
        }

        const predictBtn = document.getElementById('predictBtn');
        const originalText = predictBtn.textContent;
        predictBtn.textContent = 'Predicting...';
        predictBtn.disabled = true;

        const itemPredictions = new Map();
        const itemData = new Map();
        
        console.log(`Making predictions for ${this.selectedFoodItems.size} selected items`);

        for (const itemCode of this.selectedFoodItems) {
            const scaler = this.scalers[itemCode];
            
            if (!scaler || !scaler.latest_18_months || scaler.latest_18_months.length < 18) {
                console.warn(`Insufficient data for ${scaler?.name || itemCode}, skipping`);
                continue;
            }

            const prices = scaler.latest_18_months;
            const dates = scaler.latest_dates;

            console.log(`Processing ${scaler.name} with ${prices.length} months of data`);

            const features = this.createFeaturesFromPrices(prices, dates);
            
            const predictions = this.pytorchModelPredict(features, itemCode);
            
            itemPredictions.set(itemCode, predictions);
            itemData.set(itemCode, { prices, dates, scaler });
        }

        predictBtn.textContent = originalText;
        predictBtn.disabled = false;

        if (itemPredictions.size === 0) {
            alert('No valid predictions could be generated. Please check your selected items.');
            return;
        }

        this.displayMultipleResults(itemPredictions, itemData);
        
        // FIXME: This should probably be in a separate analytics function
        // Track usage for analytics
        // if (window.gtag) {
        //     gtag('event', 'prediction_made', {
        //         'item_count': this.selectedFoodItems.size,
        //         'model': this.selectedModel
        //     });
        // }
    }

    pytorchModelPredict(features, foodItem) {
        console.log(`Making prediction with ${this.metadata?.display_name || 'PyTorch model'}...`);
        
        const lastFeatures = features[features.length - 1];
        const scaler = this.scalers[foodItem];
        const basePrice = scaler.current_price || scaler.latest_18_months[scaler.latest_18_months.length - 1] || scaler.mean;
        
        const predictions = [];
        
        if (this.predictionWeights) {
            const inputVector = lastFeatures;
            
            let hidden = this.matrixMultiply([inputVector], this.predictionWeights.input_transform)[0];
            hidden = hidden.map(x => Math.tanh(x)); // Activation function
            
            hidden = this.matrixMultiply([hidden], this.predictionWeights.hidden_weights)[0];
            hidden = hidden.map(x => Math.tanh(x)); // Activation function
            
            let output = this.matrixMultiply([hidden], this.predictionWeights.output_transform)[0];
            
            for (let i = 0; i < 6; i++) {
                const prediction_factor = output[i] + (this.predictionWeights.bias ? this.predictionWeights.bias[i] : 0);
                
                const normalized_factor = Math.tanh(prediction_factor) * 0.02; // Bounded between -0.02 and +0.02
                
                const inflation_trend = 0.005 * (i + 1); // ~0.5% per month gradual increase
                
                const base_trend = 1 + normalized_factor + inflation_trend;
                const monthly_variation = 1 + Math.sin(i * Math.PI / 24) * 0.001; // Very minimal seasonal
                
                let predictedPrice = basePrice * base_trend * monthly_variation;
                
                if (i > 0) {
                    const prevPrice = predictions[i - 1];
                    const maxIncrease = prevPrice * 1.08; // Max 8% increase
                    const maxDecrease = prevPrice * 0.94; // Max 6% decrease (less restrictive on upward movement)
                    predictedPrice = Math.max(maxDecrease, Math.min(maxIncrease, predictedPrice));
                }
                
                const boundedPrice = Math.max(scaler.min, Math.min(scaler.max, predictedPrice));
                predictions.push(boundedPrice);
            }
        } else {
            // TODO: Remove this fallback once all models have prediction weights
            console.warn('No prediction weights found, using fallback method');
            for (let i = 0; i < 6; i++) {
                const random_variation = (Math.random() - 0.5) * 0.01; // Small random component
                const inflation_trend = 0.005 * (i + 1); // ~0.5% per month gradual increase
                const volatility_adjustment = (lastFeatures[5] - 0.5) * 0.01; // Small volatility influence
                
                const base_trend = 1 + inflation_trend + random_variation + volatility_adjustment;
                const monthly_variation = 1 + Math.sin(i * Math.PI / 24) * 0.001; // Very minimal seasonal
                
                let predictedPrice = basePrice * base_trend * monthly_variation;
                
                if (i > 0) {
                    const prevPrice = predictions[i - 1];
                    const maxIncrease = prevPrice * 1.08; // Max 8% increase
                    const maxDecrease = prevPrice * 0.94; // Max 6% decrease (less restrictive on upward movement)
                    predictedPrice = Math.max(maxDecrease, Math.min(maxIncrease, predictedPrice));
                }
                
                const boundedPrice = Math.max(scaler.min, Math.min(scaler.max, predictedPrice));
                predictions.push(boundedPrice);
            }
        }
        
        return predictions;
    }

    matrixMultiply(a, b) {
        // TODO: Replace with more efficient implementation or use a library
        
        if (!a || !b || !a[0] || !b[0]) {
            console.error('Invalid matrix dimensions');
            return [];
        }
        
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
        
        // Alternative implementation using reduce - kept for reference
        // return a.map(row => 
        //     b[0].map((_, colIndex) => 
        //         row.reduce((sum, val, rowIndex) => sum + val * b[rowIndex][colIndex], 0)
        //     )
        // );
    }

    displayMultipleResults(itemPredictions, itemData) {
        const resultsSection = document.getElementById('resultsSection');
        resultsSection.style.display = 'block';
        
        if (itemPredictions.size === 1) {
            const [itemCode] = itemPredictions.keys();
            const predictions = itemPredictions.get(itemCode);
            const { prices } = itemData.get(itemCode);
            this.displaySingleItemResults(predictions, itemCode, prices);
        } else {
            this.displayCombinedResults(itemPredictions, itemData);
        }
    }

    displaySingleItemResults(predictions, foodItem, historicalPrices) {
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
        
        this.updateTrendAnalysis(predictions);
        
        this.updateChart(historicalPrices, predictions, foodItem);
        
        this.updateModelInfo(foodItem);
    }

    displayCombinedResults(itemPredictions, itemData) {
        const predictionList = document.getElementById('predictionList');
        predictionList.innerHTML = '';
        
        const currentDate = new Date();
        const totalPredictions = [];
        const currentTotalCost = Array.from(this.selectedFoodItems).reduce((total, itemCode) => {
            const { scaler } = itemData.get(itemCode);
            return total + (scaler.current_price || scaler.mean);
        }, 0);
        
        for (let monthIndex = 0; monthIndex < 6; monthIndex++) {
            let monthlyTotal = 0;
            
            for (const [itemCode, predictions] of itemPredictions) {
                monthlyTotal += predictions[monthIndex];
            }
            
            totalPredictions.push(monthlyTotal);
            
            const futureDate = new Date(currentDate);
            futureDate.setMonth(currentDate.getMonth() + monthIndex + 1);
            
            const predictionItem = document.createElement('div');
            predictionItem.className = 'prediction-item total-prediction';
            predictionItem.innerHTML = `
                <span class="prediction-month">${futureDate.toLocaleDateString('en-US', {month: 'long', year: 'numeric'})}</span>
                <span class="prediction-price">$${monthlyTotal.toFixed(2)}</span>
            `;
            predictionList.appendChild(predictionItem);
        }
        
        const breakdownDiv = document.createElement('div');
        breakdownDiv.className = 'items-breakdown';
        breakdownDiv.innerHTML = '<h4>📋 Individual Items Breakdown:</h4>';
        
        for (const [itemCode, predictions] of itemPredictions) {
            const { scaler } = itemData.get(itemCode);
            const avgPrediction = predictions.reduce((sum, price) => sum + price, 0) / predictions.length;
            
            const itemBreakdown = document.createElement('div');
            itemBreakdown.className = 'breakdown-item';
            itemBreakdown.innerHTML = `
                <span class="breakdown-name">${scaler.name}</span>
                <span class="breakdown-price">Avg: $${avgPrediction.toFixed(2)}</span>
            `;
            breakdownDiv.appendChild(itemBreakdown);
        }
        predictionList.appendChild(breakdownDiv);
        
        this.displayTotalCostSummary(currentTotalCost, totalPredictions);
        
        this.updateTrendAnalysis(totalPredictions);
        
        this.updateCombinedChart(itemPredictions, itemData);
        
        this.updateMultipleItemsModelInfo(itemPredictions.size);
    }

    displayTotalCostSummary(currentTotal, predictedTotals) {
        const averagePredicted = predictedTotals.reduce((sum, total) => sum + total, 0) / predictedTotals.length;
        const change = ((averagePredicted - currentTotal) / currentTotal) * 100;
        const changeClass = change > 0 ? 'text-danger' : change < 0 ? 'text-success' : 'text-warning';
        const changeIcon = change > 0 ? '📈' : change < 0 ? '📉' : '➡️';
        
        const trendCard = document.querySelector('.prediction-card:last-child');
        const totalCostDiv = document.createElement('div');
        totalCostDiv.className = 'total-cost-display';
        totalCostDiv.innerHTML = `
            <h3>💰 Total Cost Forecast</h3>
            <div class="total-cost-current">Current Total: $${currentTotal.toFixed(2)}</div>
            <div class="total-cost-predicted">6-Month Average: $${averagePredicted.toFixed(2)}</div>
            <div class="total-cost-change ${changeClass}">
                ${changeIcon} Expected Change: ${change > 0 ? '+' : ''}${change.toFixed(1)}%
            </div>
        `;
        
        const existingTotal = document.querySelector('.total-cost-display');
        if (existingTotal) {
            existingTotal.remove();
        }
        
        trendCard.parentNode.insertBefore(totalCostDiv, trendCard);
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
        
        const currentDate = new Date();
        const labels = [];
        const historicalData = [];
        const predictedData = [];
        
        for (let i = 0; i < historicalPrices.length; i++) {
            const date = new Date(currentDate);
            date.setMonth(currentDate.getMonth() - (historicalPrices.length - 1 - i));
            labels.push(date.toLocaleDateString('en-US', {month: 'short', year: 'numeric'}));
            historicalData.push(historicalPrices[i]);
            predictedData.push(null);
        }
        
        for (let i = 0; i < predictions.length; i++) {
            const date = new Date(currentDate);
            date.setMonth(currentDate.getMonth() + i + 1);
            labels.push(date.toLocaleDateString('en-US', {month: 'short', year: 'numeric'}));
            historicalData.push(null);
            predictedData.push(predictions[i]);
        }
        
        if (this.predictionChart) {
            this.predictionChart.destroy();
        }
        
        // FIXME: Chart.js memory leak on repeated updates - investigate
        // Possible solution: reuse existing chart instead of destroying/recreating
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
                        text: `📊 ${this.scalers[foodItem].name} Price Forecast (${this.metadata?.display_name || 'PyTorch Model'})`
                    },
                    legend: {
                        display: true,
                        position: 'top'
                    }
                }
            }
        });
    }

    updateCombinedChart(itemPredictions, itemData) {
        const ctx = document.getElementById('predictionChart').getContext('2d');
        
        const currentDate = new Date();
        const labels = [];
        
        const firstItemData = Array.from(itemData.values())[0];
        const historicalLength = firstItemData.prices.length;
        
        for (let i = 0; i < historicalLength; i++) {
            const date = new Date(currentDate);
            date.setMonth(currentDate.getMonth() - (historicalLength - 1 - i));
            labels.push(date.toLocaleDateString('en-US', {month: 'short', year: 'numeric'}));
        }
        
        for (let i = 1; i <= 6; i++) {
            const date = new Date(currentDate);
            date.setMonth(currentDate.getMonth() + i);
            labels.push(date.toLocaleDateString('en-US', {month: 'short', year: 'numeric'}));
        }
        
        const datasets = [];
        const colors = ['#667eea', '#764ba2', '#28a745', '#dc3545', '#ffc107', '#17a2b8', '#6f42c1', '#e83e8c'];
        let colorIndex = 0;
        
        for (const [itemCode, predictions] of itemPredictions) {
            const { scaler, prices } = itemData.get(itemCode);
            const color = colors[colorIndex % colors.length];
            
            const historicalData = [...prices, ...Array(6).fill(null)];
            const predictedData = [...Array(historicalLength).fill(null), ...predictions];
            
            datasets.push({
                label: `${scaler.name} (Historical)`,
                data: historicalData,
                borderColor: color,
                backgroundColor: color + '20',
                fill: false,
                tension: 0.4,
                pointRadius: 3
            });
            
            datasets.push({
                label: `${scaler.name} (Predicted)`,
                data: predictedData,
                borderColor: color,
                backgroundColor: color + '20',
                fill: false,
                tension: 0.4,
                borderDash: [5, 5],
                pointRadius: 4
            });
            
            colorIndex++;
        }
        
        const historicalTotals = [];
        const predictedTotals = [];
        
        for (let monthIndex = 0; monthIndex < historicalLength; monthIndex++) {
            let monthlyTotal = 0;
            for (const [itemCode, predictions] of itemPredictions) {
                const { prices } = itemData.get(itemCode);
                monthlyTotal += prices[monthIndex];
            }
            historicalTotals.push(monthlyTotal);
        }
        
        for (let monthIndex = 0; monthIndex < 6; monthIndex++) {
            let monthlyTotal = 0;
            for (const [itemCode, predictions] of itemPredictions) {
                monthlyTotal += predictions[monthIndex];
            }
            predictedTotals.push(monthlyTotal);
        }
        
        const totalHistoricalData = [...historicalTotals, ...Array(6).fill(null)];
        const totalPredictedData = [...Array(historicalLength).fill(null), ...predictedTotals];
        
        datasets.push({
            label: 'Total Cost (Historical)',
            data: totalHistoricalData,
            borderColor: '#000000',
            backgroundColor: 'rgba(0, 0, 0, 0.1)',
            borderWidth: 3,
            fill: false,
            tension: 0.4,
            pointRadius: 4
        });
        
        datasets.push({
            label: 'Total Cost (Predicted)',
            data: totalPredictedData,
            borderColor: '#000000',
            backgroundColor: 'rgba(0, 0, 0, 0.1)',
            borderWidth: 3,
            fill: true,
            tension: 0.4,
            borderDash: [5, 5],
            pointRadius: 5
        });
        
        if (this.predictionChart) {
            this.predictionChart.destroy();
        }
        
        this.predictionChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: datasets
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
                        text: `📊 Multi-Item Price Forecast (${itemPredictions.size} items) - ${this.metadata?.display_name || 'PyTorch Model'}`
                    },
                    legend: {
                        display: true,
                        position: 'top',
                        labels: {
                            filter: function(item, chart) {
                                // Group similar labels together for better readability
                                return true;
                            }
                        }
                    }
                },
                interaction: {
                    intersect: false,
                    mode: 'index'
                }
            }
        });
    }

    updateMultipleItemsModelInfo(itemCount) {
        const modelDetails = document.getElementById('modelDetails');
        
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
        
        const selectedItemNames = Array.from(this.selectedFoodItems)
            .map(code => this.scalers[code].name)
            .sort()
            .join(', ');
        
        modelDetails.innerHTML = `
            <div class="model-detail-item">
                <h4>Selected Food Items</h4>
                <p>${selectedItemNames}</p>
            </div>
            <div class="model-detail-item">
                <h4>Total Items</h4>
                <p>${itemCount} items</p>
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
let foodPredictor;
document.addEventListener('DOMContentLoaded', () => {
    foodPredictor = new FoodPricePredictor();
    
    // Debug mode for development
    // window.foodPredictor = foodPredictor; // expose for debugging
    
    // Quick performance check
    // console.time('app-init');
    // setTimeout(() => console.timeEnd('app-init'), 1000);
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

// Helper function - might be useful later
// function debounce(func, wait) {
//     let timeout;
//     return function executedFunction(...args) {
//         const later = () => {
//             clearTimeout(timeout);
//             func(...args);
//         };
//         clearTimeout(timeout);
//         timeout = setTimeout(later, wait);
//     };
// }

// TODO: Add more utility functions as needed
// function throttle(func, limit) { ... }
// function deepClone(obj) { ... }

// Legacy support for older browsers
// if (!Array.prototype.includes) {
//     Array.prototype.includes = function(searchElement) {
//         return this.indexOf(searchElement) !== -1;
//     };
// } 