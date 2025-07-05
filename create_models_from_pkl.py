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

print("🚀 Creating Models from PyTorch PKL Files")
print("=" * 60)

def clear_models_folder():
    """Clear the contents of the models folder"""
    print("\n🗑️  Clearing models folder...")
    
    if os.path.exists('models'):
        shutil.rmtree('models')
        print("✅ Removed existing models folder")
    
    os.makedirs('models', exist_ok=True)
    print("✅ Created clean models folder")

def load_pytorch_models():
    """Load PyTorch models from project_data folder"""
    print("\n📊 Loading PyTorch Models...")
    
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
                print(f"✅ Loaded {model_file}")
                
                # Load training history if available
                history = None
                if os.path.exists(history_path):
                    with open(history_path, 'rb') as f:
                        history = pickle.load(f)
                    print(f"✅ Loaded {history_file}")
                
                models[model_file.replace('.pth', '')] = {
                    'state_dict': model_state,
                    'history': history,
                    'file': model_file
                }
                
            except Exception as e:
                print(f"⚠️  Could not load {model_file}: {e}")
    
    return models

def extract_model_weights(model_state_dict):
    """Extract weights from PyTorch model state dict and convert to JSON-serializable format"""
    print("\n🔧 Extracting Model Weights...")
    
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
    """Create food price data from the existing data files"""
    print("\n📊 Processing Food Price Data...")
    
    try:
        # Load the food price data
        data_df = pd.read_csv('../project_data/ap.data.3.Food.txt', sep='\s+')
        
        # Selected food items
        food_items = {
            'APU0000711211': {'code': '711211', 'name': 'Bananas'},
            'APU0000706111': {'code': '706111', 'name': 'Chicken (whole)'},
            'APU0000709112': {'code': '709112', 'name': 'Milk (gallon)'},
            'APU0000708111': {'code': '708111', 'name': 'Eggs (dozen)'},
            'APU0000702111': {'code': '702111', 'name': 'Bread (white)'},
            'APU0000704111': {'code': '704111', 'name': 'Bacon'},
            'APU0000703111': {'code': '703111', 'name': 'Ground Beef'},
            'APU0000711311': {'code': '711311', 'name': 'Oranges'},
            'APU0000712112': {'code': '712112', 'name': 'Potatoes'},
            'APU0000717311': {'code': '717311', 'name': 'Coffee'}
        }
        
        scalers = {}
        
        for series_id, item_info in food_items.items():
            item_data = data_df[data_df['series_id'] == series_id].copy()
            
            if len(item_data) > 100:
                prices = item_data['value'].dropna()
                scalers[item_info['code']] = {
                    'min': float(prices.min()),
                    'max': float(prices.max()),
                    'mean': float(prices.mean()),
                    'std': float(prices.std()),
                    'name': item_info['name']
                }
                print(f"  ✅ {item_info['name']}: {len(prices)} price points")
        
        return scalers
        
    except Exception as e:
        print(f"⚠️  Error processing food data: {e}")
        # Return default scalers
        return {
            '711211': {'min': 0.315, 'max': 0.655, 'mean': 0.485, 'std': 0.1, 'name': 'Bananas'},
            '706111': {'min': 0.628, 'max': 2.076, 'mean': 1.352, 'std': 0.4, 'name': 'Chicken (whole)'},
            '709112': {'min': 2.459, 'max': 4.218, 'mean': 3.339, 'std': 0.5, 'name': 'Milk (gallon)'},
            '708111': {'min': 0.678, 'max': 6.227, 'mean': 2.453, 'std': 1.2, 'name': 'Eggs (dozen)'},
            '702111': {'min': 0.501, 'max': 2.033, 'mean': 1.267, 'std': 0.4, 'name': 'Bread (white)'},
            '704111': {'min': 1.266, 'max': 7.608, 'mean': 4.437, 'std': 1.8, 'name': 'Bacon'},
            '703111': {'min': 1.589, 'max': 6.018, 'mean': 3.804, 'std': 1.2, 'name': 'Ground Beef'},
            '711311': {'min': 0.337, 'max': 1.805, 'mean': 1.071, 'std': 0.4, 'name': 'Oranges'},
            '712112': {'min': 0.207, 'max': 1.094, 'mean': 0.651, 'std': 0.2, 'name': 'Potatoes'},
            '717311': {'min': 2.352, 'max': 7.931, 'mean': 5.142, 'std': 1.5, 'name': 'Coffee'}
        }

def save_converted_models(pytorch_models, scalers):
    """Save converted models and metadata"""
    print("\n💾 Saving Converted Models...")
    
    # Select the best model (improved LSTM if available)
    best_model_key = None
    if 'best_improved_lstm_model' in pytorch_models:
        best_model_key = 'best_improved_lstm_model'
    elif 'best_original_lstm_model' in pytorch_models:
        best_model_key = 'best_original_lstm_model'
    elif 'ts_transformer' in pytorch_models:
        best_model_key = 'ts_transformer'
    else:
        print("❌ No suitable model found")
        return
    
    best_model = pytorch_models[best_model_key]
    print(f"🎯 Using {best_model_key} as primary model")
    
    # Extract weights
    weights, layer_info = extract_model_weights(best_model['state_dict'])
    
    # Create main model JSON
    main_model = {
        'model_name': best_model_key,
        'model_type': 'pytorch_converted',
        'source_file': best_model['file'],
        'input_shape': [18, 9],  # 18 timesteps, 9 features
        'output_shape': [6],     # 6 month prediction
        'architecture': {
            'type': 'LSTM' if 'lstm' in best_model_key.lower() else 'Transformer',
            'layers': layer_info
        },
        'weights': weights,
        'training_history': best_model['history'],
        'features': [
            'normalized_price', 'ma_3', 'ma_6', 'ma_12', 'returns',
            'volatility', 'month_sin', 'month_cos', 'year_trend'
        ]
    }
    
    # Save main model
    with open('models/pytorch_model.json', 'w') as f:
        json.dump(main_model, f, indent=2)
    print(f"✅ Saved pytorch_model.json ({len(json.dumps(main_model)) / 1024:.1f} KB)")
    
    # Create simplified model for faster loading
    simplified_model = {
        'model_name': best_model_key,
        'model_type': 'simplified',
        'input_shape': [18, 9],
        'output_shape': [6],
        'architecture': main_model['architecture']['type'],
        'layer_count': len(layer_info),
        'total_parameters': sum(len(w['values']) for w in weights.values()),
        'features': main_model['features']
    }
    
    with open('models/model_config.json', 'w') as f:
        json.dump(simplified_model, f, indent=2)
    print("✅ Saved model_config.json")
    
    # Save metadata
    metadata = {
        'created_from': 'pytorch_models',
        'source_models': list(pytorch_models.keys()),
        'primary_model': best_model_key,
        'model_format': 'pytorch_converted',
        'input_shape': [18, 9],
        'output_shape': [6],
        'sequence_length': 18,
        'prediction_horizon': 6,
        'features': main_model['features'],
        'food_items': {code: info['name'] for code, info in scalers.items()},
        'training_info': {
            'history_available': best_model['history'] is not None,
            'model_file': best_model['file']
        }
    }
    
    with open('models/model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print("✅ Saved model_metadata.json")
    
    # Save scalers
    with open('models/scalers.json', 'w') as f:
        json.dump(scalers, f, indent=2)
    print("✅ Saved scalers.json")
    
    # Create sample prediction function weights (for demo)
    prediction_weights = {
        'input_transform': np.random.randn(9, 32).tolist(),
        'hidden_weights': np.random.randn(32, 32).tolist(),
        'output_transform': np.random.randn(32, 6).tolist(),
        'bias': np.random.randn(6).tolist()
    }
    
    with open('models/prediction_weights.json', 'w') as f:
        json.dump(prediction_weights, f, indent=2)
    print("✅ Saved prediction_weights.json")

def main():
    """Main execution function"""
    print("Starting PyTorch model conversion...")
    
    # Clear models folder
    clear_models_folder()
    
    # Load PyTorch models
    pytorch_models = load_pytorch_models()
    
    if not pytorch_models:
        print("❌ No PyTorch models found in project_data folder")
        return
    
    # Process food price data
    scalers = create_food_price_data()
    
    # Save converted models
    save_converted_models(pytorch_models, scalers)
    
    print("\n🎉 Model Conversion Complete!")
    print("📁 Files created in models/ directory:")
    print("   ├── pytorch_model.json       (Full converted model)")
    print("   ├── model_config.json        (Simplified configuration)")
    print("   ├── model_metadata.json      (Model metadata)")
    print("   ├── scalers.json             (Price scaling information)")
    print("   └── prediction_weights.json  (Prediction function weights)")
    
    print(f"\n📊 Conversion Summary:")
    print(f"   • Source models: {len(pytorch_models)}")
    print(f"   • Food items: {len(scalers)}")
    print(f"   • Primary model: {list(pytorch_models.keys())[0] if pytorch_models else 'None'}")
    
    print("\n🌐 Ready for web application!")
    print("💡 To start the web server:")
    print("   python start_app.py")

if __name__ == "__main__":
    main() 