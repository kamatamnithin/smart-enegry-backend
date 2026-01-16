#!/usr/bin/env python3
"""
Test script for Energy Consumption ML API
Run this locally before deployment to verify everything works
"""

import json
import sys
import os

# Add current directory to path for imports
sys.path.append('.')

def test_health_endpoint():
    """Test the health endpoint"""
    print("Testing health endpoint...")
    try:
        from api.health import handler
        result = handler({}, {})

        # Parse the response
        if isinstance(result.get('body'), str):
            body = json.loads(result['body'])
        else:
            body = result['body']

        assert 'status' in body
        assert body['status'] == 'healthy'
        print("Health endpoint test passed")
        return True
    except Exception as e:
        print(f"Health endpoint test failed: {e}")
        return False

def test_predict_endpoint():
    """Test the prediction endpoint with sample data"""
    print("Testing prediction endpoint...")
    try:
        from api.predict import handler

        # Sample prediction request - features should be a list
        sample_data = {
            'features': [{
                'temperature': 25.0,
                'humidity': 60.0,
                'wind_speed': 5.0,
                'pressure': 1013.0,
                'hour': 12,
                'day_of_week': 1,
                'month': 6,
                'season': 'summer',
                'is_holiday': False,
                'building_type': 'residential',
                'floor_area': 100.0,
                'occupancy': 3,
                'appliance_count': 5,
                'lighting_count': 8,
                'renewable_energy': True,
                'energy_efficiency': 'medium'
            }]
        }

        event = {'body': json.dumps(sample_data), 'httpMethod': 'POST'}
        result = handler(event, {})

        assert 'statusCode' in result
        assert result['statusCode'] == 200

        # Parse response body
        if isinstance(result.get('body'), str):
            body = json.loads(result['body'])
        else:
            body = result['body']

        assert 'predictions' in body
        assert len(body['predictions']) > 0
        assert 'predicted' in body['predictions'][0]
        print("Prediction endpoint test passed")
        return True
    except Exception as e:
        print(f"Prediction endpoint test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_model_loading():
    """Test that the ML model loads correctly"""
    print("Testing model loading...")
    try:
        import pickle

        with open('random_forest_model.pkl', 'rb') as f:
            model_data = pickle.load(f)

        print(f"Model loaded successfully: {type(model_data.get('model', 'No model'))}")

        if 'feature_columns' in model_data:
            print(f"Feature columns found: {len(model_data['feature_columns'])} features")
            return True
        else:
            print("No feature_columns found in model")
            return False
    except Exception as e:
        print(f"Model loading test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Running Energy Consumption ML API Tests\n")

    tests = [
        test_model_loading,
        test_health_endpoint,
        test_predict_endpoint
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("All tests passed! Ready for deployment.")
        return 0
    else:
        print("Some tests failed. Please fix before deployment.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
