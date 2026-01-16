# Energy Consumption ML API Backend

This is the backend API for the Energy Consumption Dashboard, providing machine learning predictions for energy usage.

## Features

- **ML Predictions**: Random Forest model for energy consumption forecasting
- **Health Check**: API status monitoring endpoint
- **Railway Deployment**: Web service deployment with automatic scaling

## Deployment

### Railway Deployment

1. **Connect to Railway**:
   - Go to [railway.app](https://railway.app) and sign in
   - Click "New Project" → "Deploy from GitHub repo"
   - Connect your GitHub account and select the `smart-energy` repository

2. **Configure Environment**:
   - Railway will automatically detect Python and use the `Procfile`
   - The app will be available at: `https://your-project-name.up.railway.app`

3. **Environment Variables** (optional):
   - Railway automatically sets the `PORT` environment variable
   - No additional configuration needed

### Vercel Deployment (Alternative)

This backend can also be deployed on Vercel as serverless functions.

## API Endpoints

### GET /api/health
Returns API health status and model information.

### POST /api/predict
Accepts energy consumption features and returns ML predictions.

**Request Body:**
```json
{
  "features": [
    {
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
  ]
}
```

## Technologies

- Python 3.9
- Flask
- Scikit-learn
- Railway (web deployment)
- Gunicorn (WSGI server)
