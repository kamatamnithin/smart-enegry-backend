import pickle
from datetime import datetime
import json

# Global model variable
MODEL_PATH = 'random_forest_model.pkl'
model = None

def load_model():
    global model
    try:
        with open(MODEL_PATH, 'rb') as f:
            model_data = pickle.load(f)
        if isinstance(model_data, dict) and 'model' in model_data:
            model = model_data['model']
        else:
            model = model_data
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

# Load model on import
model_loaded = load_model()

def handler(event, context):
    """Vercel serverless handler for health check"""
    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type'
        },
        'body': json.dumps({
            'status': 'healthy',
            'model_loaded': model is not None,
            'timestamp': datetime.now().isoformat()
        })
    }