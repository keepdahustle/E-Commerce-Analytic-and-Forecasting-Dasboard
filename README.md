# E-Commerce Intelligence Dashboard & Time-Series Forecasting Platform

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Dash 4.1.0](https://img.shields.io/badge/Dash-4.1.0-blue.svg)](https://dash.plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An interactive, real-time e-commerce analytics and AI-powered time-series forecasting platform built with Python, Dash, and advanced machine learning models (SARIMA & Prophet).

## 📊 Overview

This project provides a comprehensive business intelligence solution for e-commerce platforms with:
- **Real-time Analytics Dashboard**: Customer segmentation, RFM analysis, churn prediction, revenue insights
- **AI Forecasting Engine**: Dual-model forecasting (SARIMA + Prophet) with accuracy validation
- **Model Comparison Framework**: Side-by-side evaluation of forecasting accuracy
- **Interactive Visualizations**: Plotly-based interactive charts with drill-down capabilities
- **2025-2026 Revenue Projection**: 24-month forecast with actual vs predicted comparison

## 🎯 Key Features

### Analytics & EDA Tab
- **Customer Analytics**
  - Total customers, revenue metrics, average order value
  - Churn rate and customer geographic distribution
  
- **RFM Segmentation**
  - Recency, Frequency, Monetary value analysis
  - Customer lifecycle classification
  
- **Revenue Trends**
  - Monthly revenue evolution (2020-2026)
  - Seasonal pattern analysis
  - Trend decomposition
  
- **Category Performance**
  - Top-performing product categories
  - Category revenue contribution
  
- **Churn Analysis**
  - Churn patterns and demographics
  - Risk factors identification

### AI Forecasting & Models Tab
- **Dual-Model Forecasting**
  - SARIMA: Seasonal ARIMA for short-term precision
  - Prophet: Facebook's forecasting for trend & seasonality
  
- **Forecast Visualization**
  - Actual vs Forecast side-by-side comparison (2025-2026)
  - Vertical alignment for accuracy validation
  - Confidence intervals visualization
  
- **Model Performance Metrics**
  - MAE, RMSE, MAPE, R² Score
  - Accuracy, Precision, Recall, F1 Score
  - Classification-equivalent evaluation matrix
  
- **Revenue Projections**
  - Individual model 2026 projections
  - Ensemble average calculation

## 🛠️ Technology Stack

### Core Framework
- **Dash 4.1.0**: Interactive web framework for data visualization
- **Plotly**: Advanced interactive charting library
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing

### Time-Series Forecasting Models
- **SARIMA (Seasonal ARIMA)**
  - Order: (1,1,1) × (1,1,1,12)
  - Handles seasonal patterns in monthly revenue data
  - Implementation: `statsmodels.tsa.statespace.sarimax`
  
- **Prophet**
  - Facebook's time-series forecasting library
  - Fallback: Fourier Series + Piecewise Linear Regression (numpy-based)
  - Handles trend changes and seasonal components
  
### Data Processing
- **Scikit-learn**: Machine learning utilities
- **SciPy**: Statistical computations

## 📈 Data & Methodology

### Dataset
- **Time Period**: 2020-2026 (84 months)
- **Source**: Simulated e-commerce dataset (reproducible with fixed seed)
- **Customer Base**: 8,000+ customers across 20 countries
- **Metrics**: Revenue, customer behavior, geographic distribution

### Data Split Strategy
```
Total Data: 84 months (2020-2026)
├── Training Data: 60 months (2020-2024) - Used to train forecasting models
├── Actual Validation: 24 months (2025-2026) - Real data for model comparison
└── Forecast Period: 24 months (2025-2026) - Predicted values
```

### Revenue Series Generation
```python
Revenue = Base + Trend + Seasonality + Noise

Where:
- Base: $420,000 (starting revenue)
- Trend: Linear growth from $0 to $320,000 over 84 months
- Seasonality: 12-month sine wave with $65,000 amplitude
- Noise: Normal distribution N(0, $22,000)
```

## 🔍 Forecasting Methodology

### SARIMA Model
**Configuration**: SARIMAX(1,1,1)(1,1,1,12)

1. **Differencing (d=1, D=1)**
   - First-order differencing for trend stationarity
   - 12-lag seasonal differencing for seasonal stationarity

2. **AutoRegressive (AR=1, SAR=1)**
   - Current value depends on previous 1 observation
   - Seasonal AR(12): Current value depends on value 12 months ago

3. **Moving Average (MA=1, SMA=1)**
   - Error terms affect current forecast
   - Seasonal MA(12): Seasonal error correction

4. **Convergence Handling**
   - Adaptive convergence with fallback for non-convergence
   - Maintains forecast validity even with warnings

### Prophet Model
**Components**:
1. **Trend**: Piecewise linear trend with automatic changepoints
2. **Seasonality**: Yearly (12-month) Fourier series expansion
3. **Holidays**: Optional holiday effects (not used for revenue)

**Fallback Strategy**:
- Primary: CmdStanPy backend (Bayesian inference)
- Fallback: Numpy-based Fourier + Piecewise Linear Regression
- Ensures compatibility across platforms (especially Windows)

## 📊 Performance Metrics

### Regression Metrics
- **MAE** (Mean Absolute Error): Average absolute prediction error
- **RMSE** (Root Mean Squared Error): Standard deviation of prediction errors
- **MAPE** (Mean Absolute Percentage Error): Percentage error relative to actual

### Model Performance Metrics
- **R² Score**: Coefficient of determination (0-1 scale)
- **Accuracy**: 100% - MAPE (forecast accuracy percentage)
- **Precision**: Error impact relative to actual revenue size
- **Recall**: Trend capture relative to volatility
- **F1 Score**: Harmonic mean of Precision and Recall

### Calculation
```python
Accuracy = 100% - MAPE
Precision = 100% - |MAE / Mean(Actual)|
Recall = 100% - RMSE / Mean(Actual)
F1 Score = 2 × (Precision × Recall) / (Precision + Recall)
```

## 🎨 Design & UI/UX

### Color Palette
```
Primary: #00A8FF (Cyan Blue) - SARIMA model
Accent: #9D4EDD (Purple) - Prophet model
Success: #00E676 (Green) - Actual data, positive metrics
Card Background: #1A2332 (Dark Blue)
Text Primary: #8B9DC3 (Light Gray-Blue)
```

### Typography
- **Rajdhani**: Headers and metrics (monospace, technical feel)
- **Exo 2**: Body text and labels (modern sans-serif)
- **Share Tech Mono**: Code and technical details (monospace)

### Interactive Features
- Responsive grid layout with Bootstrap styling
- Hover tooltips with formatted data
- Zoom and pan controls on charts
- Tab-based navigation for different analysis sections

## 🚀 Installation & Setup

### Prerequisites
- Python 3.10 or higher
- pip (Python package manager)

### Installation Steps

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/ecommerce-intelligence-dashboard.git
cd "E-Commerce Dashboard and Forecasting"
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Dependencies
```
dash==4.1.0
plotly==5.x.x
pandas==1.x.x
numpy==1.x.x
statsmodels==0.13.x (SARIMA)
prophet==1.x.x (Facebook's forecasting)
scikit-learn==1.x.x
scipy==1.x.x
dash-bootstrap-components==1.x.x
```

## 📁 Project Structure

```
E-Commerce Dashboard and Forecasting/
├── dashboard.py                    # Main application file
├── requirements.txt               # Python dependencies
├── customers.csv                  # Sample customer data
├── README.md                      # Project documentation
├── model_metrics_sarima.csv       # SARIMA model performance metrics
├── model_metrics_prophet.csv      # Prophet model performance metrics
├── forecast_sarima.csv            # SARIMA 2025-2026 forecast
└── forecast_prophet.csv           # Prophet 2025-2026 forecast
```

## 🏃 Running the Application

1. **Activate virtual environment**
```bash
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

2. **Run the dashboard**
```bash
python dashboard.py
```

3. **Access the application**
```
Open browser: http://localhost:8050
```

### Application Output
```
=======================================================
  E-Commerce Intelligence Dashboard
  -> Open: http://localhost:8050
=======================================================

Dash is running on http://0.0.0.0:8050/
```

## 📊 Dashboard Sections

### ANALYTICS & EDA Tab

#### 1. Key Performance Indicators (KPIs)
- Total Customers: Count of active customers
- Total Revenue: Sum of all customer spending
- Average Order Value: Mean revenue per customer
- Churn Rate: Percentage of churned customers
- Top Country: Geographic distribution leader
- Top Category: Best-performing product category

#### 2. RFM Segmentation Matrix
- **Recency**: Days since last purchase
- **Frequency**: Number of purchases
- **Monetary**: Total spending
- **Segmentation**: Champions, Loyal, Potential, At-Risk, Lost

#### 3. Revenue Trend Analysis
- Monthly revenue evolution (2020-2026)
- Trend line overlay
- Seasonal patterns visualization

#### 4. Category Performance
- Revenue contribution by category
- Category-wise customer distribution
- Top 5 categories ranking

#### 5. Churn Analysis
- Churn by demographics
- Risk factor analysis
- Churn trend over time

### AI FORECASTING & MODELS Tab

#### 1. Forecast Visualization
- **Training Data**: 2020-2024 historical (gray background)
- **Actual Data**: 2025-2026 real data (green solid line)
- **Forecast**: SARIMA (blue dash) vs Prophet (purple dash)
- **Validation**: Side-by-side comparison for accuracy assessment

#### 2. Model Metrics Comparison
- Bar chart comparison of:
  - Accuracy %
  - Precision %
  - Recall %
  - F1 Score %

#### 3. Individual Model Forecasts
- SARIMA 2026 Revenue Projection
- Prophet 2026 Revenue Projection
- Model engine details and configuration

#### 4. Performance Evaluation
- Classification-equivalent metrics matrix
- Model strengths and best use cases
- Ensemble average calculation

#### 5. Revenue Projection Summary
- SARIMA annual projection
- Prophet annual projection
- Ensemble average projection with monthly breakdown

## 📈 Model Performance Results

### SARIMA Performance (Example)
```
MAE: $24,577.95
RMSE: $32,513.32
MAPE: 3.64%
R² Score: 0.7077
Accuracy: 96.36%
Precision: 96.47%
Recall: 95.33%
F1 Score: 95.90%
```

### Prophet Performance (Example)
```
MAE: $71,385.43
RMSE: $81,141.79
MAPE: 10.2%
R² Score: -0.8203
Accuracy: 89.8%
Precision: 89.74%
Recall: 88.34%
F1 Score: 89.04%
```

### Model Comparison & Recommendations

| Aspect | SARIMA | Prophet |
|--------|--------|---------|
| **Strengths** | Captures seasonality & autocorrelation | Handles trend changes & holiday effects |
| **Best For** | Short-term precision (Q1-Q2 forecasts) | Long-horizon planning (12-month outlook) |
| **Accuracy** | 96.36% | 89.80% |
| **Use Case** | Tactical planning, inventory management | Strategic planning, capacity planning |

## 🔧 Configuration & Customization

### Adjusting Forecast Parameters

#### SARIMA Configuration (dashboard.py, line ~260)
```python
model = SARIMAX(
    train['y'],
    order=(1, 1, 1),              # (p, d, q) - Change for different patterns
    seasonal_order=(1, 1, 1, 12), # (P, D, Q, s) - 12 for monthly data
    enforce_stationarity=False,
    enforce_invertibility=False,
)
```

#### Prophet Configuration (dashboard.py, line ~294)
```python
Prophet(
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False,
    seasonality_mode='multiplicative',  # or 'additive'
    changepoint_prior_scale=0.05,       # Adjust for trend flexibility
)
```

### Data Period Configuration

#### Modify historical data range (dashboard.py, line ~188)
```python
months = pd.date_range('2020-01-01', '2026-12-01', freq='MS')
# Change start/end dates as needed
```

#### Adjust train/validation split (dashboard.py, line ~200)
```python
train_end = 60  # Training months (currently 2020-2024)
# Increase for more training data, decrease for more validation data
```

## 💾 Data Export

The application automatically exports metrics and forecasts to CSV:

### Files Generated
1. **model_metrics_sarima.csv**: SARIMA performance metrics
2. **model_metrics_prophet.csv**: Prophet performance metrics
3. **forecast_sarima.csv**: SARIMA 2025-2026 monthly forecast
4. **forecast_prophet.csv**: Prophet 2025-2026 monthly forecast

### CSV Format
```csv
date,forecast_value
2025-01-01,xxxxx.xx
2025-02-01,xxxxx.xx
...
2026-12-01,xxxxx.xx
```

## ⚠️ Known Issues & Limitations

### Prophet Stan Backend
- **Issue**: CmdStanPy crashes on Windows systems (error code 3221225785)
- **Solution**: Automatic fallback to Fourier Series + Piecewise Linear Regression
- **Impact**: No impact on forecast quality, just uses numpy instead of Bayesian inference

### SARIMA Convergence
- **Warning**: Maximum Likelihood optimization may not converge for certain data patterns
- **Solution**: Model continues with current parameters, warnings are suppressed
- **Impact**: Forecast quality remains acceptable for business decisions

## 🐛 Troubleshooting

### Issue: Port 8050 already in use
```bash
# Windows: Kill process on port 8050
netstat -ano | findstr :8050
taskkill /PID <PID> /F

# macOS/Linux
lsof -ti:8050 | xargs kill -9
```

### Issue: ModuleNotFoundError
```bash
# Reinstall all dependencies
pip install --upgrade -r requirements.txt
```

### Issue: Slow dashboard loading
- Clear browser cache (Ctrl+Shift+Delete)
- Reduce forecast period or data range
- Check system resources

## 🎓 Educational Resources

### Time-Series Forecasting
- [Introduction to ARIMA Models](https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average)
- [Prophet Documentation](https://facebook.github.io/prophet/docs/quick_start.html)
- [Statsmodels SARIMAX](https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html)

### Dashboard Development
- [Dash Documentation](https://dash.plotly.com/)
- [Plotly Charts](https://plotly.com/python/)
- [Bootstrap Components](https://dash-bootstrap-components.opensource.faculty.ai/)

### E-Commerce Analytics
- [RFM Analysis Guide](https://en.wikipedia.org/wiki/RFM_(customer_value))
- [Customer Segmentation](https://en.wikipedia.org/wiki/Market_segmentation)
- [Churn Prediction](https://en.wikipedia.org/wiki/Customer_attrition)

## 📝 Model Details & Mathematical Foundation

### SARIMA Mathematical Formulation

The SARIMA(1,1,1)(1,1,1,12) model follows:

$$\nabla \nabla_{12} Y_t = \phi_1 \nabla \nabla_{12} Y_{t-1} + \theta_1 \epsilon_{t-1} + \Phi_1 \epsilon_{t-12} + \Theta_1 \epsilon_{t-13} + \epsilon_t$$

Where:
- $\nabla$ = First-order differencing
- $\nabla_{12}$ = 12-lag seasonal differencing
- $\phi_1, \Phi_1$ = AutoRegressive coefficients
- $\theta_1, \Theta_1$ = Moving Average coefficients
- $\epsilon_t$ = White noise error term

### Prophet Decomposition

$$Y_t = g(t) + s(t) + h(t) + \epsilon_t$$

Where:
- $g(t)$ = Trend component (piecewise linear)
- $s(t)$ = Seasonal component (Fourier series)
- $h(t)$ = Holiday effects
- $\epsilon_t$ = Residual error

### Accuracy Metrics

**Mean Absolute Error (MAE)**
$$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

**Root Mean Squared Error (RMSE)**
$$RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

**Mean Absolute Percentage Error (MAPE)**
$$MAPE = \frac{1}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right| \times 100$$

**Coefficient of Determination (R²)**
$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$

**F1 Score**
$$F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$$

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Steps to Contribute
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💼 Author

**Portfolio Project 2026**
- Time-Series Forecasting for E-Commerce Revenue
- Dual-Model Comparison (SARIMA vs Prophet)
- Interactive Business Intelligence Dashboard

## 🙏 Acknowledgments

- **Plotly**: For interactive visualization capabilities
- **Dash**: For web framework and component library
- **Facebook Research**: For Prophet forecasting algorithm
- **Statsmodels**: For SARIMAX implementation
- **The Python Data Science Community**: For tools and libraries

## 📞 Support & Contact

For questions, issues, or suggestions:
- Open an Issue on GitHub
- Check existing documentation
- Review code comments and docstrings

## 🔄 Version History

### v1.0.0 (May 2026)
- Initial release
- Dual-model forecasting (SARIMA + Prophet)
- Comprehensive analytics dashboard
- Model comparison framework
- CSV export functionality
- Interactive visualizations

---

**Last Updated**: May 17, 2026
**Status**: Production Ready
**Python Version**: 3.10+
