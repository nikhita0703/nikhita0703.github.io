#!/usr/bin/env python3
"""
Create Models from PyTorch PKL Files
This script loads PyTorch models from project_data folder, clears the models folder,
and converts them to JSON format for the web application.
"""

import os
import json
import pickle
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

print("Creating Models from PyTorch PKL Files")
print("=" * 60)

def clear_models_folder():
    """Clear the contents of the models folder"""
    print("\nClearing models folder...")
    
    if os.path.exists('models'):
        shutil.rmtree('models')
        print("Removed existing models folder")
    
    os.makedirs('models', exist_ok=True)
    print("Created clean models folder")

def load_pytorch_models():
    """Load PyTorch models from project_data folder"""
    print("\nLoading PyTorch Models...")
    
    models = {}
    
    # Try to load different model files
    model_files = [
        'best_improved_lstm_model.pth',
        'best_original_lstm_model.pth',
        'ts_transformer.pth'
    ]
    
    history_files = [
        'best_improved_lstm_model_history.pkl',
        'best_original_lstm_model_history.pkl',
        'ts_transformer_history.pkl'
    ]
    
    for model_file, history_file in zip(model_files, history_files):
        model_path = f'../project_data/{model_file}'
        history_path = f'../project_data/{history_file}'
        
        if os.path.exists(model_path):
            try:
                # Load model state dict
                model_state = torch.load(model_path, map_location='cpu')
                print(f"Loaded {model_file}")
                
                # Load training history if available
                history = None
                if os.path.exists(history_path):
                    with open(history_path, 'rb') as f:
                        history = pickle.load(f)
                    print(f"Loaded {history_file}")
                
                models[model_file.replace('.pth', '')] = {
                    'state_dict': model_state,
                    'history': history,
                    'file': model_file
                }
                
            except Exception as e:
                print(f"Could not load {model_file}: {e}")
    
    return models

def extract_model_weights(model_state_dict):
    """Extract weights from PyTorch model state dict and convert to JSON-serializable format"""
    print("\nExtracting Model Weights...")
    
    weights = {}
    layer_info = []
    
    for name, param in model_state_dict.items():
        if isinstance(param, torch.Tensor):
            # Convert tensor to numpy array and then to list for JSON serialization
            weight_array = param.detach().cpu().numpy()
            weights[name] = {
                'shape': list(weight_array.shape),
                'values': weight_array.flatten().tolist(),
                'dtype': str(weight_array.dtype)
            }
            
            # Extract layer information
            layer_info.append({
                'name': name,
                'shape': list(weight_array.shape),
                'type': 'weight' if 'weight' in name else 'bias',
                'layer': name.split('.')[0] if '.' in name else name
            })
    
    return weights, layer_info

def create_food_price_data():
    """Create food price data from all available items in the real data files"""
    print("\nProcessing Real Food Price Data...")
    
    # Extended list of known good food items from the data
    known_food_items = {
        # Grains & Bread
        '701111': 'Flour (white)',
        '701312': 'Rice (white, uncooked)',
        '701322': 'Spaghetti and macaroni',
        '702111': 'Bread (white)',
        '702212': 'Bread (whole wheat)',
        '702421': 'Cookies (chocolate chip)',
        
        # Meat & Poultry
        '703111': 'Ground chuck (100% beef)',
        '703112': 'Ground beef (100% beef)',
        '703113': 'Ground beef (lean)',
        '703213': 'Chuck roast (boneless)',
        '703311': 'Round roast (boneless)',
        '703432': 'Beef for stew',
        '703511': 'Steak (round, boneless)',
        '703613': 'Steak (sirloin, boneless)',
        '704111': 'Bacon (sliced)',
        '704211': 'Pork chops (center cut)',
        '704212': 'Pork chops (boneless)',
        '704312': 'Ham (boneless)',
        '706111': 'Chicken (whole)',
        '706212': 'Chicken legs',
        '708111': 'Eggs (grade A, large)',
        
        # Dairy
        '709112': 'Milk (whole, gallon)',
        '710211': 'American cheese',
        '710212': 'Cheddar cheese',
        '710411': 'Ice cream',
        
        # Fruits & Vegetables
        '711211': 'Bananas',
        '711311': 'Oranges (Navel)',
        '711411': 'Grapefruit',
        '711412': 'Lemons',
        '711414': 'Peaches',
        '711415': 'Strawberries',
        '712112': 'Potatoes (white)',
        '712211': 'Lettuce (iceberg)',
        '712311': 'Tomatoes (field grown)',
        '712406': 'Peppers (sweet)',
        '712412': 'Broccoli',
        
        # Pantry Items
        '713111': 'Orange juice (frozen)',
        '714221': 'Corn (canned)',
        '714233': 'Beans (dried)',
        '715211': 'Sugar (white)',
        '715212': 'Sugar (white, packaged)',
        '717311': 'Coffee (ground roast)',
        '718311': 'Potato chips',
        
        # Beverages & Other
        '720111': 'Beer (malt beverages)',
        '720222': 'Vodka',
        '720311': 'Wine (table wine)',
        
        # Aggregate Categories
        'FC1101': 'All ground beef',
        'FC2101': 'All beef roasts',
        'FC3101': 'All beef steaks',
        'FD2101': 'All ham',
        'FD3101': 'All pork chops',
        'FF1101': 'Chicken breast (boneless)',
        'FJ1101': 'Milk (low-fat/skim)',
        'FJ4101': 'Yogurt',
        'FL2101': 'Lettuce (romaine)',
        'FN1101': 'Soft drinks (2 liter)',
        'FN1102': 'Soft drinks (12-pack)',
        'FS1101': 'Butter (stick)'
    }
    
    print(f"  Processing {len(known_food_items)} known food items...")
    
    try:
        # Load the data files using simple pandas approach for known items
        data_df = pd.read_csv('../project_data/ap.data.3.Food.txt', sep='\s+')
        
        scalers = {}
        processed_items = 0
        failed_items = 0
        
        for item_code, item_name in known_food_items.items():
            try:
                # Look for this item in the data with U.S. average series
                series_id = f'APU0000{item_code}'
                item_data = data_df[data_df['series_id'] == series_id].copy()
                
                if len(item_data) >= 18:  # Need at least 18 months of data
                    # Clean and validate data
                    item_data['value'] = pd.to_numeric(item_data['value'], errors='coerce')
                    item_data = item_data.dropna(subset=['value'])
                    
                    # Sort by year and period
                    item_data['period_num'] = item_data['period'].str.replace('M', '').astype(int)
                    item_data = item_data.sort_values(['year', 'period_num'])
                    
                    prices = item_data['value']
                    
                    # Validate price data quality
                    if (len(prices) >= 18 and 
                        prices.min() > 0 and 
                        prices.max() < 1000 and 
                        prices.mean() > 0.01):
                        
                        scalers[item_code] = {
                            'min': float(prices.min()),
                            'max': float(prices.max()),
                            'mean': float(prices.mean()),
                            'std': float(prices.std()),
                            'name': item_name,
                            'data_points': len(prices),
                            'series_id': series_id,
                            'start_year': int(item_data['year'].min()),
                            'end_year': int(item_data['year'].max())
                        }
                        processed_items += 1
                        print(f"  {item_name}: {len(prices)} points (${prices.mean():.2f} avg)")
                    else:
                        failed_items += 1
                        print(f"  {item_name}: insufficient or invalid data")
                else:
                    failed_items += 1
                    print(f"  {item_name}: not enough data points ({len(item_data)})")
                    
            except Exception as e:
                failed_items += 1
                print(f"  {item_name}: {e}")
                continue
        
        print(f"\nSuccessfully processed {processed_items} food items!")
        print(f"Failed to process {failed_items} items")
        
        # If we got some real data, return it, otherwise fall back to defaults
        if processed_items > 0:
            return scalers
        else:
            raise Exception("No items processed successfully")
        
    except Exception as e:
        print(f"Error processing food data: {e}")
        print("  Using fallback food items...")
        # Return enhanced default scalers with more variety
        return {
            # Original 10 items
            '711211': {'min': 0.315, 'max': 0.655, 'mean': 0.485, 'std': 0.1, 'name': 'Bananas', 'data_points': 544, 'series_id': 'APU0000711211', 'start_year': 1980, 'end_year': 2025},
            '706111': {'min': 0.628, 'max': 2.076, 'mean': 1.352, 'std': 0.4, 'name': 'Chicken (whole)', 'data_points': 544, 'series_id': 'APU0000706111', 'start_year': 1980, 'end_year': 2025},
            '709112': {'min': 2.459, 'max': 4.218, 'mean': 3.339, 'std': 0.5, 'name': 'Milk (whole, gallon)', 'data_points': 365, 'series_id': 'APU0000709112', 'start_year': 1995, 'end_year': 2025},
            '708111': {'min': 0.678, 'max': 6.227, 'mean': 2.453, 'std': 1.2, 'name': 'Eggs (grade A, large)', 'data_points': 544, 'series_id': 'APU0000708111', 'start_year': 1980, 'end_year': 2025},
            '702111': {'min': 0.501, 'max': 2.033, 'mean': 1.267, 'std': 0.4, 'name': 'Bread (white)', 'data_points': 544, 'series_id': 'APU0000702111', 'start_year': 1980, 'end_year': 2025},
            '704111': {'min': 1.266, 'max': 7.608, 'mean': 4.437, 'std': 1.8, 'name': 'Bacon (sliced)', 'data_points': 544, 'series_id': 'APU0000704111', 'start_year': 1980, 'end_year': 2025},
            '703111': {'min': 1.589, 'max': 6.018, 'mean': 3.804, 'std': 1.2, 'name': 'Ground chuck (100% beef)', 'data_points': 544, 'series_id': 'APU0000703111', 'start_year': 1980, 'end_year': 2025},
            '711311': {'min': 0.337, 'max': 1.805, 'mean': 1.071, 'std': 0.4, 'name': 'Oranges (Navel)', 'data_points': 544, 'series_id': 'APU0000711311', 'start_year': 1980, 'end_year': 2025},
            '712112': {'min': 0.207, 'max': 1.094, 'mean': 0.651, 'std': 0.2, 'name': 'Potatoes (white)', 'data_points': 470, 'series_id': 'APU0000712112', 'start_year': 1986, 'end_year': 2025},
            '717311': {'min': 2.352, 'max': 7.931, 'mean': 5.142, 'std': 1.5, 'name': 'Coffee (ground roast)', 'data_points': 544, 'series_id': 'APU0000717311', 'start_year': 1980, 'end_year': 2025},
            
            # Additional items for variety
            '701312': {'min': 0.85, 'max': 1.65, 'mean': 1.25, 'std': 0.2, 'name': 'Rice (white, uncooked)', 'data_points': 400, 'series_id': 'APU0000701312', 'start_year': 1980, 'end_year': 2025},
            '703112': {'min': 2.1, 'max': 5.8, 'mean': 3.95, 'std': 1.0, 'name': 'Ground beef (100% beef)', 'data_points': 490, 'series_id': 'APU0000703112', 'start_year': 1984, 'end_year': 2025},
            '704212': {'min': 2.8, 'max': 8.5, 'mean': 5.65, 'std': 1.5, 'name': 'Pork chops (boneless)', 'data_points': 360, 'series_id': 'APU0000704212', 'start_year': 1995, 'end_year': 2025},
            '710212': {'min': 3.2, 'max': 7.8, 'mean': 5.5, 'std': 1.2, 'name': 'Cheddar cheese', 'data_points': 490, 'series_id': 'APU0000710212', 'start_year': 1984, 'end_year': 2025},
            '711412': {'min': 0.9, 'max': 2.1, 'mean': 1.5, 'std': 0.3, 'name': 'Lemons', 'data_points': 544, 'series_id': 'APU0000711412', 'start_year': 1980, 'end_year': 2025},
            '712311': {'min': 0.8, 'max': 3.2, 'mean': 2.0, 'std': 0.6, 'name': 'Tomatoes (field grown)', 'data_points': 544, 'series_id': 'APU0000712311', 'start_year': 1980, 'end_year': 2025},
            '715211': {'min': 0.35, 'max': 1.2, 'mean': 0.75, 'std': 0.2, 'name': 'Sugar (white)', 'data_points': 544, 'series_id': 'APU0000715211', 'start_year': 1980, 'end_year': 2025},
            '718311': {'min': 2.8, 'max': 6.5, 'mean': 4.65, 'std': 1.0, 'name': 'Potato chips', 'data_points': 544, 'series_id': 'APU0000718311', 'start_year': 1980, 'end_year': 2025},
            'FF1101': {'min': 2.5, 'max': 6.2, 'mean': 4.35, 'std': 1.0, 'name': 'Chicken breast (boneless)', 'data_points': 230, 'series_id': 'APU0000FF1101', 'start_year': 2006, 'end_year': 2025},
            'FJ4101': {'min': 0.8, 'max': 1.8, 'mean': 1.3, 'std': 0.25, 'name': 'Yogurt', 'data_points': 85, 'series_id': 'APU0000FJ4101', 'start_year': 2018, 'end_year': 2025}
        }

def save_converted_models(pytorch_models, scalers):
    """Save converted models and metadata for each model in separate subfolders"""
    print("\nSaving All Models to Separate Subfolders...")
    
    available_models = []
    
    # Common features for all models
    common_features = [
        'normalized_price', 'ma_3', 'ma_6', 'ma_12', 'returns',
        'volatility', 'month_sin', 'month_cos', 'year_trend'
    ]
    
    # Load historical data for web application
    enhanced_scalers = load_historical_data_for_web(scalers)
    
    # Process each model
    for model_key, model_data in pytorch_models.items():
        print(f"\nProcessing {model_key}...")
        
        # Create subfolder for this model
        model_folder = f'models/{model_key}'
        os.makedirs(model_folder, exist_ok=True)
        
        # Extract weights
        weights, layer_info = extract_model_weights(model_data['state_dict'])
        
        # Determine model type
        if 'lstm' in model_key.lower():
            model_type = 'LSTM'
            display_name = model_key.replace('_', ' ').title()
        elif 'transformer' in model_key.lower():
            model_type = 'Transformer'
            display_name = 'Time Series Transformer'
        else:
            model_type = 'Neural Network'
            display_name = model_key.replace('_', ' ').title()
        
        # Create main model JSON
        main_model = {
            'model_name': model_key,
            'model_type': 'pytorch_converted',
            'source_file': model_data['file'],
            'input_shape': [18, 9],  # 18 timesteps, 9 features
            'output_shape': [6],     # 6 month prediction
            'architecture': {
                'type': model_type,
                'layers': layer_info
            },
            'weights': weights,
            'training_history': model_data['history'],
            'features': common_features
        }
        
        # Save main model
        model_file = f'{model_folder}/pytorch_model.json'
        with open(model_file, 'w') as f:
            json.dump(main_model, f, indent=2)
        
        model_size_kb = len(json.dumps(main_model)) / 1024
        print(f"  Saved {model_key}/pytorch_model.json ({model_size_kb:.1f} KB)")
        
        # Create simplified model config
        simplified_model = {
            'model_name': model_key,
            'model_type': 'simplified',
            'input_shape': [18, 9],
            'output_shape': [6],
            'architecture': model_type,
            'layer_count': len(layer_info),
            'total_parameters': sum(len(w['values']) for w in weights.values()),
            'features': common_features
        }
        
        with open(f'{model_folder}/model_config.json', 'w') as f:
            json.dump(simplified_model, f, indent=2)
        print(f"  Saved {model_key}/model_config.json")
        
        # Save model-specific metadata
        metadata = {
            'created_from': 'pytorch_models',
            'model_name': model_key,
            'display_name': display_name,
            'model_format': 'pytorch_converted',
            'architecture_type': model_type,
            'input_shape': [18, 9],
            'output_shape': [6],
            'sequence_length': 18,
            'prediction_horizon': 6,
            'features': common_features,
            'food_items': {code: info['name'] for code, info in enhanced_scalers.items()},
            'training_info': {
                'history_available': model_data['history'] is not None,
                'model_file': model_data['file'],
                'total_parameters': simplified_model['total_parameters'],
                'layer_count': len(layer_info)
            }
        }
        
        with open(f'{model_folder}/model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  Saved {model_key}/model_metadata.json")
        
        # Save enhanced scalers with historical data
        with open(f'{model_folder}/scalers.json', 'w') as f:
            json.dump(enhanced_scalers, f, indent=2)
        print(f"  Saved {model_key}/scalers.json")
        
        # Create sample prediction function weights (for demo)
        prediction_weights = {
            'input_transform': np.random.randn(9, 32).tolist(),
            'hidden_weights': np.random.randn(32, 32).tolist(),
            'output_transform': np.random.randn(32, 6).tolist(),
            'bias': np.random.randn(6).tolist()
        }
        
        with open(f'{model_folder}/prediction_weights.json', 'w') as f:
            json.dump(prediction_weights, f, indent=2)
        print(f"  Saved {model_key}/prediction_weights.json")
        
        # Add to available models list
        available_models.append({
            'id': model_key,
            'name': display_name,
            'type': model_type,
            'folder': model_key,
            'parameters': simplified_model['total_parameters'],
            'layers': len(layer_info),
            'has_history': model_data['history'] is not None,
            'source_file': model_data['file']
        })
    
    # Create models index file
    models_index = {
        'available_models': available_models,
        'default_model': available_models[0]['id'] if available_models else None,
        'total_models': len(available_models),
        'features': common_features,
        'input_shape': [18, 9],
        'output_shape': [6],
        'food_items': {code: info['name'] for code, info in enhanced_scalers.items()}
    }
    
    with open('models/models_index.json', 'w') as f:
        json.dump(models_index, f, indent=2)
    print(f"\nSaved models_index.json with {len(available_models)} available models")

def load_historical_data_for_web(scalers):
    """Load actual historical price sequences for web application"""
    print("\nLoading historical price sequences for web application...")
    
    enhanced_scalers = {}
    
    try:
        # Load the raw data
        data_df = pd.read_csv('../project_data/ap.data.3.Food.txt', sep='\s+')
        
        for item_code, scaler_info in scalers.items():
            try:
                # Get the series ID
                series_id = scaler_info['series_id']
                
                # Load the raw data for this item
                item_data = data_df[data_df['series_id'] == series_id].copy()
                
                if len(item_data) >= 18:
                    # Clean and sort the data
                    item_data['value'] = pd.to_numeric(item_data['value'], errors='coerce')
                    item_data = item_data.dropna(subset=['value'])
                    item_data['period_num'] = item_data['period'].str.replace('M', '').astype(int)
                    item_data = item_data.sort_values(['year', 'period_num'])
                    
                    # Get the most recent 24 months of data (we'll use last 18 for input)
                    recent_data = item_data.tail(24).copy()
                    
                    # Create price history arrays
                    prices = recent_data['value'].tolist()
                    years = recent_data['year'].tolist()
                    periods = recent_data['period'].tolist()
                    
                    # Create date strings for web display
                    dates = []
                    for year, period in zip(years, periods):
                        month = int(period.replace('M', ''))
                        try:
                            date_obj = pd.Timestamp(year=year, month=month, day=1)
                            dates.append(date_obj.strftime('%Y-%m'))
                        except:
                            dates.append(f"{year}-{period}")
                    
                    # Copy original scaler info and add historical data
                    enhanced_info = scaler_info.copy()
                    enhanced_info['historical_prices'] = prices
                    enhanced_info['historical_dates'] = dates
                    enhanced_info['latest_18_months'] = prices[-18:] if len(prices) >= 18 else prices
                    enhanced_info['latest_dates'] = dates[-18:] if len(dates) >= 18 else dates
                    enhanced_info['current_price'] = prices[-1] if prices else enhanced_info['mean']
                    
                    enhanced_scalers[item_code] = enhanced_info
                    print(f"  Loaded {len(prices)} months of data for {scaler_info['name']}")
                    
                else:
                    # Fallback to original scaler with synthetic data
                    enhanced_info = scaler_info.copy()
                    # Generate synthetic 18 months of data based on statistics
                    base_price = scaler_info['mean']
                    synthetic_prices = []
                    current_date = pd.Timestamp.now()
                    
                    for i in range(18):
                        # Add some realistic variation
                        variation = np.random.normal(0, scaler_info['std'] * 0.1)
                        seasonal_factor = 1 + 0.05 * np.sin(i * np.pi / 6)  # Semi-annual cycle
                        price = base_price * seasonal_factor + variation
                        price = max(scaler_info['min'], min(scaler_info['max'], price))
                        synthetic_prices.append(round(price, 3))
                    
                    synthetic_dates = []
                    for i in range(18):
                        date = current_date - pd.DateOffset(months=17-i)
                        synthetic_dates.append(date.strftime('%Y-%m'))
                    
                    enhanced_info['historical_prices'] = synthetic_prices
                    enhanced_info['historical_dates'] = synthetic_dates
                    enhanced_info['latest_18_months'] = synthetic_prices
                    enhanced_info['latest_dates'] = synthetic_dates
                    enhanced_info['current_price'] = synthetic_prices[-1]
                    enhanced_info['data_source'] = 'synthetic'
                    
                    enhanced_scalers[item_code] = enhanced_info
                    print(f"  Generated synthetic data for {scaler_info['name']}")
                    
            except Exception as e:
                print(f"  Error loading data for {scaler_info['name']}: {e}")
                # Use the original scaler as fallback
                enhanced_scalers[item_code] = scaler_info
                
        print(f"Successfully enhanced {len(enhanced_scalers)} food items with historical data")
        return enhanced_scalers
        
    except Exception as e:
        print(f"Error loading historical data: {e}")
        print("Using original scalers without historical sequences")
        return scalers

def main():
    """Main execution function"""
    print("Starting PyTorch model conversion...")
    
    # Clear models folder
    clear_models_folder()
    
    # Load PyTorch models
    pytorch_models = load_pytorch_models()
    
    if not pytorch_models:
        print("No PyTorch models found in project_data folder")
        return
    
    # Process food price data
    scalers = create_food_price_data()
    
    # Save converted models
    save_converted_models(pytorch_models, scalers)
    
    print("\nModel Conversion Complete!")
    print("Files created in models/ directory:")
    print("   ├── models_index.json        (Available models list)")
    for model_key in pytorch_models.keys():
        print(f"   ├── {model_key}/")
        print(f"   │   ├── pytorch_model.json")
        print(f"   │   ├── model_config.json")
        print(f"   │   ├── model_metadata.json")
        print(f"   │   ├── scalers.json")
        print(f"   │   └── prediction_weights.json")
    
    print(f"\nConversion Summary:")
    print(f"   • Source models: {len(pytorch_models)}")
    print(f"   • Food items: {len(scalers)}")
    print(f"   • Available models: {', '.join(pytorch_models.keys())}")
    
    print("\nReady for web application!")
    print("To start the web server:")
    print("   python start_app.py")

if __name__ == "__main__":
    main() 