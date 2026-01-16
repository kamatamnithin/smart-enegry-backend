# Energy Consumption ML API Backend

This is the backend API for the Energy Consumption Dashboard, providing machine learning predictions for energy usage.

## Features

- **ML Predictions**: Random Forest model for energy consumption forecasting
- **Health Check**: API status monitoring endpoint
- **Serverless**: Deployed on Vercel for automatic scaling

## API Endpoints

### GET /api/health
Returns API health status and model information.

### POST /api/predict
Accepts energy consumption features and returns ML predictions.

**Request Body:**
```json
{
  "features": {
    "temperature": 25.5,
    "humidity": 60.0,
    "wind_speed": 5.2,
    "pressure": 1013.25,
    "hour": 14,
    "day_of_week": 2,
    "month": 6,
    "season": "summer",
    "is_holiday": false,
    "building_type": "residential",
    "floor_area": 150.0,
    "occupancy": 4,
    "appliance_count": 8,
    "lighting_count": 12,
    "renewable_energy": true,
    "energy_efficiency": "high"
  }
}
```

## Deployment

This backend is deployed on Vercel and connected to the React frontend dashboard.

## Technologies

- Python 3.11
- Flask
- Scikit-learn
- Vercel (serverless)
