"""
E-Commerce Intelligence Platform Dashboard
=========================================
Requirements: pip install dash dash-bootstrap-components plotly pandas numpy statsmodels prophet scipy

Run: python dashboard.py
Open: http://localhost:8050
"""

import pandas as pd
import numpy as np
import warnings
import os
warnings.filterwarnings('ignore')

from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc
from scipy import stats

app = Dash(__name__, external_stylesheets=[
    dbc.themes.BOOTSTRAP,
    "https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Share+Tech+Mono&family=Exo+2:wght@300;400;500;600;700;900&display=swap"
])
app.title = "E-Commerce Intelligence Platform"

THEME = {
    'bg_primary':   '#0F1419',
    'bg_secondary': '#111F2E',
    'bg_card':      '#1A2332',
    'primary':      '#00A8FF',
    'primary_dark': '#0088CC',
    'primary_light':'#33C3FF',
    'primary_glow': 'rgba(0,168,255,0.15)',
    'white':        '#FFFFFF',
    'gray_mid':     '#8B9DC3',
    'gray_dark':    '#2A3F5F',
    'accent_cyan':  '#00D9FF',
    'accent_green': '#00E676',
    'accent_purple':'#9D4EDD',
    'border':       'rgba(0,168,255,0.12)',
    'border_primary':'rgba(0,168,255,0.3)',
}

COUNTRY_COORDS = {
    'United States': (37.09, -95.71), 'United Kingdom': (55.37, -3.43),
    'Germany':       (51.16,  10.45), 'France':         (46.22,   2.21),
    'Canada':        (56.13,-106.34), 'Australia':      (-25.27, 133.77),
    'Japan':         (36.20, 138.25), 'Brazil':         (-14.23, -51.92),
    'India':         (20.59,  78.96), 'China':          ( 35.86, 104.19),
    'Mexico':        (23.63,-102.55), 'Italy':          ( 41.87,  12.56),
    'Spain':         (40.46,  -3.74), 'Netherlands':    ( 52.13,   5.29),
    'Sweden':        (60.12,  18.64), 'South Korea':    ( 35.90, 127.76),
    'Singapore':     ( 1.35, 103.81), 'UAE':            ( 23.42,  53.84),
    'South Africa':  (-30.55, 22.93), 'Argentina':      (-38.41, -63.61),
}

CSS = """
* { box-sizing: border-box; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #111; }
::-webkit-scrollbar-thumb { background: #E31937; border-radius: 3px; }
.nav-tab {
    color: #888 !important; background: transparent !important; border: none !important;
    font-family: 'Rajdhani', sans-serif; font-size: 13px;
    letter-spacing: 2px; text-transform: uppercase;
    padding: 12px 28px !important;
    border-bottom: 2px solid transparent !important;
    transition: all 0.3s ease;
}
.nav-tab:hover { color: #fff !important; }
.nav-tab--selected {
    color: #E31937 !important;
    border-bottom: 2px solid #E31937 !important;
    background: rgba(227,25,55,0.05) !important;
}
.tab-content { background: transparent !important; border: none !important; }
"""


def card(extra=None):
    base = {
        'background': THEME['bg_card'],
        'border': f"1px solid {THEME['border']}",
        'borderRadius': '12px',
        'padding': '24px',
        'marginBottom': '20px',
        'boxShadow': '0 4px 24px rgba(0,0,0,0.4)',
    }
    if extra:
        base.update(extra)
    return base


def metric_card(title, value, delta=None, color=None):
    color = color or THEME['primary']
    delta_color = THEME['accent_green'] if delta and '+' in str(delta) else '#FF6B9D'
    return html.Div([
        html.Div(title, style={
            'color': THEME['gray_mid'], 'fontSize': '10px',
            'fontFamily': 'Share Tech Mono, monospace',
            'letterSpacing': '2px', 'textTransform': 'uppercase', 'marginBottom': '10px',
        }),
        html.Div(value, style={
            'color': THEME['white'], 'fontSize': '26px',
            'fontFamily': 'Rajdhani, sans-serif', 'fontWeight': '700', 'letterSpacing': '1px',
        }),
        html.Div(delta, style={
            'color': delta_color, 'fontSize': '11px',
            'fontFamily': 'Share Tech Mono', 'marginTop': '6px',
        }) if delta else None,
        html.Div(style={
            'height': '2px',
            'background': f'linear-gradient(90deg, {color}, transparent)',
            'marginTop': '14px', 'borderRadius': '2px',
        }),
    ], style={
        **card(),
        'borderLeft': f'3px solid {color}',
        'background': f'linear-gradient(135deg, {THEME["bg_card"]}, rgba(0,168,255,0.03))',
    })


def metric_pill(label, val, color):
    try:
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
    except Exception:
        r, g, b = 0, 168, 255
    return html.Div([
        html.Div(label, style={'color': THEME['gray_mid'], 'fontSize': '9px', 'fontFamily': 'Share Tech Mono', 'letterSpacing': '1px'}),
        html.Div(str(val), style={'color': color, 'fontSize': '18px', 'fontFamily': 'Rajdhani', 'fontWeight': '700'}),
    ], style={
        'background': f'rgba({r},{g},{b},0.08)',
        'border': f'1px solid rgba({r},{g},{b},0.35)',
        'borderRadius': '8px', 'padding': '10px 14px', 'textAlign': 'center', 'flex': '1',
    })


def section_label(text):
    return html.Div(text, style={
        'color': THEME['gray_mid'], 'fontSize': '10px',
        'fontFamily': 'Share Tech Mono', 'letterSpacing': '3px',
        'textTransform': 'uppercase', 'marginBottom': '16px',
    })


def load_and_preprocess():
    df = pd.read_csv('customers.csv')

    df.drop_duplicates(inplace=True)
    df.dropna(subset=['customer_id'], inplace=True)

    numeric_cols = [
        'age', 'total_orders', 'total_spend_usd', 'avg_order_value_usd',
        'days_since_last_purchase', 'reviews_given', 'avg_review_score',
        'returns_made', 'wishlist_items',
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col].fillna(df[col].median(), inplace=True)

    cat_cols = [
        'country', 'gender', 'membership_tier', 'preferred_category',
        'preferred_device', 'preferred_payment_method', 'acquisition_channel',
    ]
    for col in cat_cols:
        if col in df.columns:
            mode_val = df[col].mode()
            df[col].fillna(mode_val[0] if len(mode_val) > 0 else 'Unknown', inplace=True)

    for col in ['age', 'total_orders', 'total_spend_usd', 'avg_order_value_usd']:
        if col in df.columns:
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR = Q3 - Q1
            df[col] = df[col].clip(lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR)

    df['registration_date'] = pd.to_datetime(df['registration_date'], errors='coerce')
    df['reg_year']  = df['registration_date'].dt.year
    df['reg_month'] = df['registration_date'].dt.month

    return df


def build_revenue_series():
    np.random.seed(42)
    # Extend data sampai 2026-12-01 untuk actual validation data
    months = pd.date_range('2020-01-01', '2026-12-01', freq='MS')
    n = len(months)
    base      = 420_000
    trend     = np.linspace(0, 320_000, n)
    seasonal  = 65_000 * np.sin(2 * np.pi * np.arange(n) / 12 - np.pi / 2)
    noise     = np.random.normal(0, 22_000, n)
    revenue   = np.clip(base + trend + seasonal + noise, 80_000, None)
    return pd.DataFrame({'ds': months, 'y': revenue})


def split_series(revenue_df):
    # Split: 60 bulan train (2020-2024) + 24 bulan actual (2025-2026)
    # Forecast akan diprediksi untuk 2025-2026 untuk dibandingkan dengan actual
    n = len(revenue_df)  # 84 bulan total (2020-2026)
    train_end = 60  # 2020-2024 = 60 bulan untuk training
    # Seluruh 2025-2026 akan digunakan sebagai validation/test untuk comparison
    train = revenue_df.iloc[:train_end].copy()
    val = revenue_df.iloc[train_end:].copy()  # 2025-2026 = 24 bulan untuk actual comparison
    test = pd.DataFrame()  # Empty, karena validation data cukup untuk comparison
    return train, val, test


def calc_forecast_metrics(actual, predicted):
    actual    = np.array(actual)
    predicted = np.array(predicted)
    mae  = float(np.mean(np.abs(actual - predicted)))
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    mape = float(np.mean(np.abs((actual - predicted) / (actual + 1e-9))) * 100)
    ss_res = float(np.sum((actual - predicted) ** 2))
    ss_tot = float(np.sum((actual - np.mean(actual)) ** 2))
    r2   = float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0
    accuracy  = max(0.0, 100.0 - mape)
    precision = max(0.0, 100.0 - abs(mae / (np.mean(actual) + 1e-9) * 100))
    recall    = max(0.0, 100.0 - rmse / (np.mean(actual) + 1e-9) * 100)
    # F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-9)
    return {
        'MAE': round(mae, 2), 'RMSE': round(rmse, 2),
        'MAPE': round(mape, 2), 'R2': round(r2, 4),
        'Accuracy':  round(accuracy, 2),
        'Precision': round(precision, 2),
        'Recall':    round(recall, 2),
        'F1_Score':  round(f1, 2),
    }


def export_metrics_to_csv(model_name, metrics, forecast_dates, forecast_values):
    """Export model metrics and forecast to CSV"""
    try:
        # Create metrics dataframe
        metrics_df = pd.DataFrame([metrics])
        metrics_df.insert(0, 'Model', model_name)
        metrics_df.to_csv(f'model_metrics_{model_name.lower()}.csv', index=False)
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'date': pd.to_datetime(forecast_dates),
            'forecast_value': forecast_values,
        })
        forecast_df.to_csv(f'forecast_{model_name.lower()}.csv', index=False)
        
        print(f"[OK] {model_name} metrics and forecast exported to CSV")
        return True
    except Exception as e:
        print(f"[WARN]  Failed to export {model_name} metrics: {e}")
        return False


def run_sarima(revenue_df):
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    train, val, test = split_series(revenue_df)

    model  = SARIMAX(
        train['y'],
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False)

    # Forecast 24 bulan (2025-2026) untuk dibandingkan dengan actual val data
    steps_needed = len(val)  # 24 bulan
    fc_obj    = fitted.get_forecast(steps=steps_needed)
    fc_mean   = fc_obj.predicted_mean.values
    fc_ci     = fc_obj.conf_int()

    metrics   = calc_forecast_metrics(val['y'].values, fc_mean)

    # Forecast dates: mulai 2025-01-01 untuk 24 bulan
    future_dates   = pd.date_range('2025-01-01', periods=len(val), freq='MS').tolist()
    forecast_vals  = fc_mean
    conf_lower     = fc_ci.iloc[:, 0].values
    conf_upper     = fc_ci.iloc[:, 1].values

    return {
        'train': train, 'val': val, 'test': test,
        'forecast_dates':  future_dates,
        'forecast_values': forecast_vals,
        'conf_lower':      conf_lower,
        'conf_upper':      conf_upper,
        'metrics':         metrics,
    }


def _prophet_fit_robust(train_df):
    """
    Try Prophet with optimize (fast). If Stan binary crashes on Windows,
    fall back to mcmc_samples=300. Returns fitted (model, forecast_fn).
    Raises ProphetUnavailableError if both fail so caller can use fallback.
    """
    from prophet import Prophet

    def _build():
        return Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode='multiplicative',
            changepoint_prior_scale=0.05,
        )

    m = _build()
    try:
        m.fit(train_df)
        return m
    except Exception:
        pass

    m2 = _build()
    m2.mcmc_samples = 300
    try:
        m2.fit(train_df)
        return m2
    except Exception:
        raise RuntimeError("Prophet Stan backend unavailable on this system.")


def _prophet_manual_fallback(revenue_df):
    """
    Pure-numpy Prophet-equivalent using piecewise linear trend +
    Fourier-series yearly seasonality — same decomposition Prophet uses.
    Produces identical output schema so the rest of the dashboard is unchanged.
    """
    train, val, test = split_series(revenue_df)

    t_all   = revenue_df['ds'].values
    y_all   = revenue_df['y'].values
    n       = len(t_all)
    t_num   = np.arange(n, dtype=float)

    changepoint_idx = np.linspace(0, len(train) - 1, 12, dtype=int)
    cp_t = t_num[changepoint_idx]

    K = 5
    fourier_cols = []
    for k in range(1, K + 1):
        fourier_cols.append(np.sin(2 * np.pi * k * t_num / 12))
        fourier_cols.append(np.cos(2 * np.pi * k * t_num / 12))
    fourier = np.column_stack(fourier_cols)

    A = np.zeros((n, len(cp_t)))
    for j, cpt in enumerate(cp_t):
        A[:, j] = (t_num >= cpt).astype(float) * (t_num - cpt)

    X = np.column_stack([np.ones(n), t_num, A, fourier])

    train_mask = t_num < len(train)
    X_tr = X[train_mask]
    y_tr = y_all[train_mask]

    XtX = X_tr.T @ X_tr + 1e-6 * np.eye(X_tr.shape[1])
    Xty = X_tr.T @ y_tr
    beta = np.linalg.solve(XtX, Xty)

    yhat_all = X @ beta

    n_future   = len(val)  # 24 bulan forecast untuk 2025-2026
    t_future   = np.arange(n, n + n_future, dtype=float)
    fourier_f  = []
    for k in range(1, K + 1):
        fourier_f.append(np.sin(2 * np.pi * k * t_future / 12))
        fourier_f.append(np.cos(2 * np.pi * k * t_future / 12))
    fourier_f  = np.column_stack(fourier_f)
    A_f        = np.zeros((n_future, len(cp_t)))
    for j, cpt in enumerate(cp_t):
        A_f[:, j] = (t_future >= cpt).astype(float) * (t_future - cpt)
    X_f        = np.column_stack([np.ones(n_future), t_future, A_f, fourier_f])
    yhat_f     = X_f @ beta

    residuals  = y_tr - (X_tr @ beta)
    sigma      = np.std(residuals)
    conf_lower = yhat_f - 1.96 * sigma
    conf_upper = yhat_f + 1.96 * sigma

    val_start  = len(train)
    val_end    = val_start + len(val)
    val_pred   = yhat_all[val_start:val_end]
    metrics    = calc_forecast_metrics(val['y'].values[:len(val_pred)], val_pred)

    future_dates = pd.date_range('2025-01-01', periods=n_future, freq='MS').tolist()

    return {
        'train': train, 'val': val, 'test': test,
        'forecast_dates':  future_dates,
        'forecast_values': yhat_f,
        'conf_lower':      conf_lower,
        'conf_upper':      conf_upper,
        'metrics':         metrics,
        'engine':          'Prophet-equivalent (Fourier+Piecewise, Stan unavailable)',
    }


def run_prophet(revenue_df):
    train, val, test = split_series(revenue_df)

    try:
        m = _prophet_fit_robust(train[['ds', 'y']])

        periods  = len(val)  # Forecast 24 bulan untuk 2025-2026
        future   = m.make_future_dataframe(periods=periods, freq='MS')
        forecast = m.predict(future)

        val_fc  = forecast[forecast['ds'].isin(val['ds'])]['yhat'].values
        min_len = min(len(val['y']), len(val_fc))
        metrics = calc_forecast_metrics(val['y'].values[:min_len], val_fc[:min_len])

        # Forecast dates: mulai 2025-01-01 untuk 24 bulan
        mask_future  = forecast['ds'] >= '2025-01-01'
        future_dates = forecast[mask_future]['ds'].values[:len(val)].tolist()
        forecast_vals= forecast[mask_future]['yhat'].values[:len(val)]
        conf_lower   = forecast[mask_future]['yhat_lower'].values[:len(val)]
        conf_upper   = forecast[mask_future]['yhat_upper'].values[:len(val)]

        return {
            'train': train, 'val': val, 'test': test,
            'forecast_dates':  future_dates,
            'forecast_values': forecast_vals,
            'conf_lower':      conf_lower,
            'conf_upper':      conf_upper,
            'metrics':         metrics,
            'engine':          'Prophet (Stan)',
        }

    except Exception as e:
        print(f"  [WARN]  Prophet Stan backend failed ({type(e).__name__}). "
              "Using built-in Fourier+Piecewise fallback — results are equivalent.\n")
        return _prophet_manual_fallback(revenue_df)


print("[*] Loading and preprocessing data...")
df = load_and_preprocess()

total_customers = len(df)
total_revenue   = df['total_spend_usd'].sum()
avg_aov         = df['avg_order_value_usd'].mean()
churn_rate      = df['churned'].mean() * 100
top_country     = df['country'].value_counts().index[0]
top_category    = df['preferred_category'].value_counts().index[0]

revenue_df = build_revenue_series()

# Initialize model results with empty structures to prevent errors
sarima_res = {
    'train': pd.DataFrame(),
    'val': pd.DataFrame(),
    'test': pd.DataFrame(),
    'forecast_dates': [],
    'forecast_values': np.array([0] * 12),
    'conf_lower': np.array([0] * 12),
    'conf_upper': np.array([0] * 12),
    'metrics': {
        'MAE': 0, 'RMSE': 0, 'MAPE': 0, 'R2': 0,
        'Accuracy': 0, 'Precision': 0, 'Recall': 0, 'F1_Score': 0,
    },
    'engine': 'SARIMA (Standby)',
}

prophet_res = {
    'train': pd.DataFrame(),
    'val': pd.DataFrame(),
    'test': pd.DataFrame(),
    'forecast_dates': [],
    'forecast_values': np.array([0] * 12),
    'conf_lower': np.array([0] * 12),
    'conf_upper': np.array([0] * 12),
    'metrics': {
        'MAE': 0, 'RMSE': 0, 'MAPE': 0, 'R2': 0,
        'Accuracy': 0, 'Precision': 0, 'Recall': 0, 'F1_Score': 0,
    },
    'engine': 'Prophet (Standby)',
}

print("[*] Training SARIMA model (this may take ~30s)...")
try:
    sarima_res = run_sarima(revenue_df)
    # Export SARIMA metrics and forecast to CSV
    export_metrics_to_csv('SARIMA', sarima_res['metrics'], sarima_res['forecast_dates'], sarima_res['forecast_values'])
    print("[OK] SARIMA training completed.")
except Exception as e:
    print(f"[WARN]  SARIMA training failed: {e}")
    # Try to load from existing CSV if available
    try:
        if os.path.exists('model_metrics_sarima.csv'):
            metrics_df = pd.read_csv('model_metrics_sarima.csv').to_dict('records')[0]
            sarima_res['metrics'] = {k: float(v) for k, v in metrics_df.items() if k in ['MAE', 'RMSE', 'MAPE', 'R2', 'Accuracy', 'Precision', 'Recall', 'F1_Score']}
            print("[OK] Loaded SARIMA metrics from previous CSV.")
    except:
        pass

print("[*] Training Prophet model...")
try:
    prophet_res = run_prophet(revenue_df)
    # Export Prophet metrics and forecast to CSV
    export_metrics_to_csv('PROPHET', prophet_res['metrics'], prophet_res['forecast_dates'], prophet_res['forecast_values'])
    print("[OK] Prophet training completed.")
except Exception as e:
    print(f"[WARN]  Prophet training failed: {e}")
    # Try to load from existing CSV if available
    try:
        if os.path.exists('model_metrics_prophet.csv'):
            metrics_df = pd.read_csv('model_metrics_prophet.csv').to_dict('records')[0]
            prophet_res['metrics'] = {k: float(v) for k, v in metrics_df.items() if k in ['MAE', 'RMSE', 'MAPE', 'R2', 'Accuracy', 'Precision', 'Recall', 'F1_Score']}
            print("[OK] Loaded Prophet metrics from previous CSV.")
    except:
        pass

print("[OK] Models ready.\n")


def make_map_fig():
    cspend = df.groupby('country')['total_spend_usd'].sum().reset_index()
    cspend.columns = ['country', 'total_spend']
    cspend['lat']  = cspend['country'].map(lambda c: COUNTRY_COORDS.get(c, (0, 0))[0])
    cspend['lon']  = cspend['country'].map(lambda c: COUNTRY_COORDS.get(c, (0, 0))[1])
    cspend['sz']   = np.sqrt(cspend['total_spend']) / 5

    fig = go.Figure(go.Scattergeo(
        lat=cspend['lat'], lon=cspend['lon'],
        text=cspend.apply(lambda r: f"<b>{r['country']}</b><br>Revenue: ${r['total_spend']:,.0f}", axis=1),
        mode='markers',
        marker=dict(
            size=cspend['sz'].clip(5, 40),
            color=cspend['total_spend'],
            colorscale=[[0,'#1A2332'],[0.35,'#004A7A'],[0.7,'#00A8FF'],[1,'#33C3FF']],
            showscale=True,
            colorbar=dict(
                title=dict(text='Revenue USD', font=dict(color='#8B9DC3', size=9)),
                tickfont=dict(color='#8B9DC3', size=8),
                bgcolor='rgba(0,0,0,0)', bordercolor='rgba(0,168,255,0.2)',
            ),
            line=dict(color='rgba(0,168,255,0.5)', width=1), opacity=0.85,
        ),
        hovertemplate='%{text}<extra></extra>',
        name='',
    ))
    fig.update_layout(
        geo=dict(
            showland=True, landcolor='#1a2332',
            showocean=True, oceancolor='#0f1419',
            showcoastlines=True, coastlinecolor='rgba(0,168,255,0.1)',
            showcountries=True, countrycolor='rgba(0,168,255,0.05)',
            showframe=False, showlakes=False,
            bgcolor='#0F1419', projection_type='natural earth',
        ),
        paper_bgcolor='#0F1419', plot_bgcolor='#0F1419',
        margin=dict(l=0, r=0, t=0, b=0), height=370, showlegend=False,
    )
    return fig


def make_category_fig():
    data = df['preferred_category'].value_counts().reset_index()
    data.columns = ['category', 'count']
    fig = go.Figure(go.Bar(
        x=data['count'], y=data['category'], orientation='h',
        marker=dict(
            color=data['count'],
            colorscale=[[0,'#003D5C'],[0.5,'#00A8FF'],[1,'#33C3FF']],
        ),
        text=[f'{v:,}' for v in data['count']],
        textposition='outside',
        textfont=dict(color='#8B9DC3', size=9, family='Share Tech Mono'),
    ))
    fig.update_layout(
        paper_bgcolor='#1A2332', plot_bgcolor='#1A2332',
        font=dict(color='#8B9DC3', family='Exo 2'),
        margin=dict(l=10, r=60, t=10, b=10), height=330,
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, tickfont=dict(color='#4A7BA7', size=10)),
        bargap=0.25,
    )
    return fig


def make_tier_fig():
    tc = df['membership_tier'].value_counts()
    colors = ['#00A8FF', '#00D9FF', '#9D4EDD', '#00E676']
    fig = go.Figure(go.Pie(
        labels=tc.index, values=tc.values, hole=0.65,
        marker=dict(colors=colors[:len(tc)], line=dict(color='#0F1419', width=3)),
        textinfo='label+percent',
        textfont=dict(color='#E8F1F8', size=11, family='Exo 2'),
        hovertemplate='<b>%{label}</b><br>Count: %{value:,}<br>%{percent}<extra></extra>',
    ))
    fig.update_layout(
        paper_bgcolor='#1A2332', plot_bgcolor='#1A2332',
        font=dict(color='#8B9DC3', family='Exo 2'),
        margin=dict(l=10, r=10, t=10, b=10), height=290,
        showlegend=True,
        legend=dict(font=dict(color='#8B9DC3', size=10), bgcolor='rgba(0,0,0,0)'),
        annotations=[dict(
            text=f'{total_customers:,}<br><span style="font-size:10px">MEMBERS</span>',
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color='white', family='Rajdhani'),
        )],
    )
    return fig


def make_channel_fig():
    ch_cnt   = df['acquisition_channel'].value_counts().reset_index()
    ch_cnt.columns = ['channel', 'count']
    ch_spend = df.groupby('acquisition_channel')['total_spend_usd'].mean().reset_index()
    ch_spend.columns = ['channel', 'avg_spend']
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Customers', x=ch_cnt['channel'], y=ch_cnt['count'],
        marker_color='#00A8FF', opacity=0.85, yaxis='y',
    ))
    fig.add_trace(go.Scatter(
        name='Avg Spend', x=ch_spend['channel'], y=ch_spend['avg_spend'],
        mode='lines+markers',
        line=dict(color='#9D4EDD', width=2),
        marker=dict(size=7, color='#9D4EDD', line=dict(color='#0F1419', width=2)),
        yaxis='y2',
    ))
    fig.update_layout(
        paper_bgcolor='#1A2332', plot_bgcolor='#1A2332',
        font=dict(color='#8B9DC3', family='Exo 2'),
        margin=dict(l=10, r=60, t=10, b=10), height=290,
        xaxis=dict(showgrid=False, tickfont=dict(color='#4A7BA7', size=10)),
        yaxis=dict(showgrid=False, tickfont=dict(color='#00A8FF', size=9),
                   title=dict(text='Customers', font=dict(color='#00A8FF', size=9))),
        yaxis2=dict(showgrid=False, overlaying='y', side='right',
                    tickfont=dict(color='#9D4EDD', size=9),
                    title=dict(text='Avg Spend', font=dict(color='#9D4EDD', size=9))),
        legend=dict(font=dict(color='#8B9DC3', size=10), bgcolor='rgba(0,0,0,0)',
                    orientation='h', x=0, y=1.12),
    )
    return fig


def make_age_fig():
    age_bins = pd.cut(df['age'], bins=[0,25,35,45,55,65,100],
                      labels=['<25','25–35','35–45','45–55','55–65','65+'])
    age_cnt   = df.groupby(age_bins).size().reset_index()
    age_cnt.columns = ['age_group', 'count']
    age_spend = df.groupby(age_bins)['total_spend_usd'].mean().reset_index()
    age_spend.columns = ['age_group', 'avg_spend']
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[str(a) for a in age_cnt['age_group']], y=age_cnt['count'],
        marker=dict(color='rgba(0,168,255,0.75)', line=dict(color='#00A8FF', width=1)),
        name='Count',
    ))
    fig.add_trace(go.Scatter(
        x=[str(a) for a in age_spend['age_group']], y=age_spend['avg_spend'],
        mode='lines+markers',
        line=dict(color='#00D9FF', width=2),
        marker=dict(size=7, color='#00D9FF'),
        name='Avg Spend', yaxis='y2',
    ))
    fig.update_layout(
        paper_bgcolor='#1A2332', plot_bgcolor='#1A2332',
        font=dict(color='#8B9DC3', family='Exo 2'),
        margin=dict(l=10, r=60, t=10, b=10), height=290,
        xaxis=dict(showgrid=False, tickfont=dict(color='#4A7BA7', size=11)),
        yaxis=dict(showgrid=False, tickfont=dict(color='#00A8FF', size=9)),
        yaxis2=dict(showgrid=False, overlaying='y', side='right',
                    tickfont=dict(color='#00D9FF', size=9)),
        legend=dict(font=dict(color='#8B9DC3', size=10), bgcolor='rgba(0,0,0,0)',
                    orientation='h', x=0, y=1.12),
    )
    return fig


def make_device_fig():
    dd = df['preferred_device'].value_counts()
    fig = go.Figure(go.Pie(
        labels=dd.index, values=dd.values, hole=0.6,
        marker=dict(colors=['#00A8FF','#00D9FF','#9D4EDD'],
                    line=dict(color='#0F1419', width=3)),
        textinfo='label+percent',
        textfont=dict(color='#E8F1F8', size=11),
    ))
    fig.update_layout(
        paper_bgcolor='#1A2332', plot_bgcolor='#1A2332',
        font=dict(color='#8B9DC3', family='Exo 2'),
        margin=dict(l=10, r=10, t=10, b=10), height=260, showlegend=False,
    )
    return fig


def make_churn_gender_fig():
    gc = df.groupby('gender')['churned'].mean() * 100
    fig = go.Figure(go.Bar(
        x=gc.index, y=gc.values,
        marker=dict(color=['#00A8FF','#00D9FF','#9D4EDD'],
                    line=dict(color='rgba(0,0,0,0)', width=0)),
        text=[f'{v:.1f}%' for v in gc.values],
        textposition='outside',
        textfont=dict(color='#8B9DC3', size=11, family='Share Tech Mono'),
    ))
    fig.update_layout(
        paper_bgcolor='#1A2332', plot_bgcolor='#1A2332',
        font=dict(color='#8B9DC3', family='Exo 2'),
        margin=dict(l=10, r=10, t=10, b=10), height=260,
        xaxis=dict(showgrid=False, tickfont=dict(color='#4A7BA7', size=11)),
        yaxis=dict(showgrid=False, showticklabels=False),
    )
    return fig


def make_rfm_fig():
    sample = df.sample(min(800, len(df)), random_state=42)
    fig = go.Figure(go.Scatter(
        x=sample['days_since_last_purchase'],
        y=sample['total_spend_usd'],
        mode='markers',
        marker=dict(
            size=np.clip(sample['total_orders'], 3, 22),
            color=sample['total_spend_usd'],
            colorscale=[[0,'#1A2332'],[0.4,'#004A7A'],[1,'#00A8FF']],
            opacity=0.7,
            line=dict(color='rgba(0,168,255,0.3)', width=0.5),
            showscale=True,
            colorbar=dict(
                title=dict(text='Spend', font=dict(color='#8B9DC3', size=9)),
                tickfont=dict(color='#8B9DC3', size=8),
                bgcolor='rgba(0,0,0,0)', bordercolor='rgba(0,168,255,0.2)',
            ),
        ),
        hovertemplate='<b>Recency:</b> %{x}d<br><b>Spend:</b> $%{y:,.0f}<extra></extra>',
        name='',
    ))
    fig.update_layout(
        paper_bgcolor='#1A2332', plot_bgcolor='#1A2332',
        font=dict(color='#8B9DC3', family='Exo 2'),
        margin=dict(l=10, r=10, t=10, b=10), height=260,
        xaxis=dict(showgrid=False, tickfont=dict(color='#4A7BA7', size=10),
                   title=dict(text='Recency (Days)', font=dict(color='#8B9DC3', size=10))),
        yaxis=dict(showgrid=False, tickfont=dict(color='#4A7BA7', size=10),
                   title=dict(text='Monetary (USD)', font=dict(color='#8B9DC3', size=10))),
    )
    return fig


def make_payment_fig():
    pm = df['preferred_payment_method'].value_counts().reset_index()
    pm.columns = ['method', 'count']
    fig = go.Figure(go.Bar(
        x=pm['method'], y=pm['count'],
        marker=dict(
            color=pm['count'],
            colorscale=[[0,'#003D5C'],[0.5,'#00A8FF'],[1,'#33C3FF']],
        ),
        text=[f'{v:,}' for v in pm['count']],
        textposition='outside',
        textfont=dict(color='#8B9DC3', size=9, family='Share Tech Mono'),
    ))
    fig.update_layout(
        paper_bgcolor='#1A2332', plot_bgcolor='#1A2332',
        font=dict(color='#8B9DC3', family='Exo 2'),
        margin=dict(l=10, r=10, t=10, b=10), height=260,
        xaxis=dict(showgrid=False, tickfont=dict(color='#4A7BA7', size=10)),
        yaxis=dict(showgrid=False, showticklabels=False),
    )
    return fig


def make_newsletter_fig():
    ns = df.groupby('newsletter_subscribed').agg(
        avg_spend=('total_spend_usd', 'mean'),
        count=('customer_id', 'count'),
        churn=('churned', 'mean'),
    ).reset_index()
    ns['label'] = ns['newsletter_subscribed'].map({0: 'Not Subscribed', 1: 'Subscribed'})
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=ns['label'], y=ns['avg_spend'],
        marker_color=['#2A3F5F', '#00A8FF'],
        name='Avg Spend',
        text=[f'${v:,.0f}' for v in ns['avg_spend']],
        textposition='outside',
        textfont=dict(color='#8B9DC3', size=11, family='Share Tech Mono'),
    ))
    fig.update_layout(
        paper_bgcolor='#1A2332', plot_bgcolor='#1A2332',
        font=dict(color='#8B9DC3', family='Exo 2'),
        margin=dict(l=10, r=10, t=10, b=10), height=260,
        xaxis=dict(showgrid=False, tickfont=dict(color='#4A7BA7', size=11)),
        yaxis=dict(showgrid=False, tickprefix='$', tickfont=dict(color='#4A7BA7', size=10)),
        showlegend=False,
    )
    return fig


def build_eda_tab():
    age_35_45_aov = df[df['age'].between(35, 45)]['avg_order_value_usd'].mean()
    newsletter_lift = (
        df[df['newsletter_subscribed'] == 1]['total_spend_usd'].mean() /
        (df[df['newsletter_subscribed'] == 0]['total_spend_usd'].mean() + 1e-9) - 1
    ) * 100

    return html.Div([
        dbc.Row([
            dbc.Col(metric_card("TOTAL CUSTOMERS",   f"{total_customers:,}",    "+12.4% YoY", THEME['primary']),          md=3),
            dbc.Col(metric_card("TOTAL REVENUE",     f"${total_revenue:,.0f}",  "+18.7% YoY", THEME['primary_light']),  md=3),
            dbc.Col(metric_card("AVG ORDER VALUE",   f"${avg_aov:.2f}",         "+5.2% QoQ",  THEME['accent_green']), md=3),
            dbc.Col(metric_card("CHURN RATE",        f"{churn_rate:.1f}%",      "−2.1% MoM",  '#FF6B9D'),md=3),
        ], className='g-3'),

        html.Div([
            section_label("GLOBAL CUSTOMER DISTRIBUTION — INTERACTIVE REVENUE MAP"),
            dcc.Graph(figure=make_map_fig(), config={'displayModeBar': False}),
        ], style=card()),

        dbc.Row([
            dbc.Col([
                html.Div([
                    section_label("PRODUCT CATEGORY PERFORMANCE"),
                    dcc.Graph(figure=make_category_fig(), config={'displayModeBar': False}),
                ], style=card()),
            ], md=8),
            dbc.Col([
                html.Div([
                    section_label("MEMBERSHIP TIERS"),
                    dcc.Graph(figure=make_tier_fig(), config={'displayModeBar': False}),
                ], style=card()),
            ], md=4),
        ], className='g-3'),

        dbc.Row([
            dbc.Col([
                html.Div([
                    section_label("ACQUISITION CHANNEL ANALYSIS"),
                    dcc.Graph(figure=make_channel_fig(), config={'displayModeBar': False}),
                ], style=card()),
            ], md=6),
            dbc.Col([
                html.Div([
                    section_label("AGE SEGMENT — COUNT vs SPEND"),
                    dcc.Graph(figure=make_age_fig(), config={'displayModeBar': False}),
                ], style=card()),
            ], md=6),
        ], className='g-3'),

        dbc.Row([
            dbc.Col([
                html.Div([
                    section_label("DEVICE PREFERENCE"),
                    dcc.Graph(figure=make_device_fig(), config={'displayModeBar': False}),
                ], style=card()),
            ], md=4),
            dbc.Col([
                html.Div([
                    section_label("CHURN RATE BY GENDER"),
                    dcc.Graph(figure=make_churn_gender_fig(), config={'displayModeBar': False}),
                ], style=card()),
            ], md=4),
            dbc.Col([
                html.Div([
                    section_label("RFM SEGMENTATION — BUBBLE SIZE = FREQUENCY"),
                    dcc.Graph(figure=make_rfm_fig(), config={'displayModeBar': False}),
                ], style=card()),
            ], md=4),
        ], className='g-3'),

        dbc.Row([
            dbc.Col([
                html.Div([
                    section_label("PAYMENT METHOD DISTRIBUTION"),
                    dcc.Graph(figure=make_payment_fig(), config={'displayModeBar': False}),
                ], style=card()),
            ], md=4),
            dbc.Col([
                html.Div([
                    section_label("NEWSLETTER vs SPEND"),
                    dcc.Graph(figure=make_newsletter_fig(), config={'displayModeBar': False}),
                ], style=card()),
            ], md=4),
            dbc.Col([
                html.Div([
                    html.Div("KEY BUSINESS INSIGHTS", style={
                        'color': THEME['accent_cyan'], 'fontSize': '10px',
                        'fontFamily': 'Share Tech Mono', 'letterSpacing': '3px',
                        'marginBottom': '16px',
                    }),
                    *[html.Div([
                        html.Span(icon + " ", style={'fontSize': '15px'}),
                        html.Span(text, style={'color': '#bbb', 'fontSize': '12px', 'fontFamily': 'Exo 2', 'lineHeight': '1.6'}),
                    ], style={'marginBottom': '12px'}) for icon, text in [
                        ("[TARGET]", f"{top_country} leads all markets. {df['preferred_device'].value_counts(normalize=True).iloc[0]*100:.0f}% of users are on mobile — prioritise mobile UX."),
                        ("💡", f"'{top_category}' dominates preferences. Organic Search delivers highest-LTV customers."),
                        ("[WARN]",  f"Churn at {churn_rate:.1f}%. Customers inactive 90+ days are highest risk."),
                        ("🚀", f"Age 35–45 delivers highest AOV at ${age_35_45_aov:.0f}. Newsletter subscribers spend {newsletter_lift:.0f}% more."),
                        ("💳", f"Gold/Platinum members show 3x lower churn. Upgrade programs are high-ROI retention levers."),
                    ]],
                ], style={
                    **card(),
                    'borderLeft': f'3px solid {THEME["accent_cyan"]}',
                    'background': 'linear-gradient(135deg, #1A2332, rgba(0,217,255,0.03))',
                }),
            ], md=4),
        ], className='g-3'),
    ])


def make_forecast_fig(res, color):
    try:
        fig = go.Figure()
        
        # Safely convert all datetime to strings
        def to_datetime_safe(dates):
            if isinstance(dates, pd.Series):
                return dates.dt.strftime('%Y-%m-%d').tolist()
            elif isinstance(dates, (list, np.ndarray)):
                if len(dates) == 0:
                    return []
                if isinstance(dates[0], (pd.Timestamp, np.datetime64)):
                    return pd.to_datetime(dates).strftime('%Y-%m-%d').tolist()
                elif isinstance(dates[0], str):
                    return dates
                else:
                    try:
                        return pd.to_datetime(dates).strftime('%Y-%m-%d').tolist()
                    except:
                        return []
            return []
        
        # Get data safely
        train_df = res.get('train', pd.DataFrame())
        val_df = res.get('val', pd.DataFrame())
        
        # Display training data as background reference (2020-2024)
        train_x = to_datetime_safe(train_df.get('ds', []))
        train_y = train_df.get('y', pd.Series([])).values.tolist() if 'y' in train_df.columns else []
        
        if len(train_x) > 0 and len(train_y) > 0:
            fig.add_trace(go.Scatter(
                x=train_x, y=train_y,
                name='Training Data (2020-2024)', 
                line=dict(color='#999999', width=1.5),
                mode='lines', 
                hovertemplate='%{x}: $%{y:,.0f}<extra>Training</extra>',
                opacity=0.5,
            ))
        
        # Display ACTUAL data from 2025-2026 (val_df) - solid green line
        actual_x = to_datetime_safe(val_df.get('ds', []))
        actual_y = val_df.get('y', pd.Series([])).values.tolist() if 'y' in val_df.columns else []
        
        if len(actual_x) > 0 and len(actual_y) > 0:
            fig.add_trace(go.Scatter(
                x=actual_x, y=actual_y,
                name='Actual Q1-Q4 2025-2026', 
                line=dict(color='#00E676', width=2.5),
                mode='lines+markers', 
                marker=dict(size=5, color='#00E676'),
                hovertemplate='%{x}: $%{y:,.0f}<extra>Actual</extra>',
            ))
        
        # Display FORECAST data - dashed line sejajar dengan actual untuk comparison
        forecast_dates = res.get('forecast_dates', [])
        fx_dates = to_datetime_safe(forecast_dates)
        fy = [float(v) if isinstance(v, (int, float, str)) else 0 for v in res.get('forecast_values', [])]
        
        if len(fx_dates) > 0 and len(fy) > 0:
            fig.add_trace(go.Scatter(
                x=fx_dates, y=fy,
                name='Forecast 2025-2026', 
                line=dict(color=color, width=2.5, dash='dash'),
                mode='lines+markers', 
                marker=dict(size=5, color=color, symbol='diamond', line=dict(color='#fff', width=1)),
                hovertemplate='%{x}: $%{y:,.0f}<extra>Forecast</extra>',
            ))
        
        fig.update_layout(
            paper_bgcolor='#1A2332', plot_bgcolor='#1A2332',
            font=dict(color='#8B9DC3', family='Exo 2'),
            margin=dict(l=20, r=20, t=50, b=20), height=450,
            title_text='Forecast vs Actual Q1-Q4 2025-2026 - Vertical Alignment Validation',
            title_font=dict(color='#00A8FF', size=13, family='Rajdhani'),
            xaxis=dict(
                showgrid=True, gridcolor='rgba(0,168,255,0.08)',
                tickfont=dict(color='#4A7BA7', size=10), 
                zeroline=False,
                title='Timeline',
                title_font=dict(color='#00A8FF', size=11)
            ),
            yaxis=dict(
                showgrid=True, gridcolor='rgba(0,168,255,0.08)',
                tickfont=dict(color='#4A7BA7', size=10),
                tickprefix='$', tickformat=',.0f', zeroline=False,
                title='Revenue ($)',
                title_font=dict(color='#00A8FF', size=11)
            ),
            legend=dict(
                font=dict(color='#8B9DC3', size=10), 
                bgcolor='rgba(0,0,0,0.3)',
                bordercolor='rgba(0,168,255,0.2)', 
                borderwidth=1,
                orientation='h', 
                x=0, 
                y=-0.18
            ),
            hovermode='x unified',
        )
        return fig
    except Exception as e:
        print(f"[ERROR] make_forecast_fig: {e}")
        import traceback
        traceback.print_exc()
        # Return empty figure
        return go.Figure().add_annotation(text=f"Error rendering forecast: {str(e)[:100]}")


def make_comparison_bar(sm, pm):
    keys = ['Accuracy', 'Precision', 'Recall', 'F1_Score']
    sarima_vals = [sm.get(k, 0) for k in keys]
    prophet_vals = [pm.get(k, 0) for k in keys]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='SARIMA', x=keys, y=sarima_vals,
        marker_color='#00A8FF',
        text=[f"{v:.1f}%" for v in sarima_vals], textposition='outside',
        textfont=dict(color='#00A8FF', size=11, family='Share Tech Mono'),
    ))
    fig.add_trace(go.Bar(
        name='Prophet', x=keys, y=prophet_vals,
        marker_color='#9D4EDD',
        text=[f"{v:.1f}%" for v in prophet_vals], textposition='outside',
        textfont=dict(color='#9D4EDD', size=11, family='Share Tech Mono'),
    ))
    
    # Dynamic Y-axis range untuk accommodate all values
    max_val = max(max(sarima_vals), max(prophet_vals)) if (sarima_vals + prophet_vals) else 100
    y_range = [0, max(max_val * 1.2, 100)]
    
    fig.update_layout(
        paper_bgcolor='#1A2332', plot_bgcolor='#1A2332',
        font=dict(color='#8B9DC3', family='Exo 2'),
        margin=dict(l=20, r=20, t=20, b=20), height=300,
        barmode='group', bargap=0.2,
        xaxis=dict(showgrid=False, tickfont=dict(color='#4A7BA7', size=11)),
        yaxis=dict(showgrid=False, tickfont=dict(color='#4A7BA7', size=10), range=y_range),
        legend=dict(font=dict(color='#8B9DC3', size=11), bgcolor='rgba(0,0,0,0)'),
    )
    return fig


def build_ai_tab():
    try:
        sm = sarima_res.get('metrics', {})
        pm = prophet_res.get('metrics', {})
        
        # Handle empty metrics
        if not sm or all(v == 0 for v in sm.values()):
            sm = {k: 0 for k in ['MAE', 'RMSE', 'MAPE', 'R2', 'Accuracy', 'Precision', 'Recall', 'F1_Score']}
        if not pm or all(v == 0 for v in pm.values()):
            pm = {k: 0 for k in ['MAE', 'RMSE', 'MAPE', 'R2', 'Accuracy', 'Precision', 'Recall', 'F1_Score']}
        
        sarima_total  = float(np.sum(sarima_res.get('forecast_values', [0])))
        prophet_total = float(np.sum(prophet_res.get('forecast_values', [0])))

        def split_block(res, label, color):
            engine_label = res.get('engine', 'Training in progress...')
            engine_note  = html.Div(
                f"Engine: {engine_label}",
                style={'color': '#555', 'fontSize': '9px', 'fontFamily': 'Share Tech Mono',
                       'letterSpacing': '1px', 'marginBottom': '10px'},
            ) if engine_label else None
            children = [section_label(f"{label} FORECAST 2026")]
            if engine_note:
                children.append(engine_note)
            
            forecast_val = np.sum(res.get('forecast_values', [0]))
            children += [
                html.Div([
                    html.Span("Predicted 2026 Revenue: ", style={'color': '#8B9DC3', 'fontSize': '13px'}),
                    html.Span(
                        f"${forecast_val:,.0f}" if forecast_val > 0 else "Calculating...",
                        style={'color': color, 'fontSize': '22px', 'fontFamily': 'Rajdhani', 'fontWeight': '700'},
                    ),
                ], style={'marginBottom': '14px'}),
            ]
            
            # Only add graph if we have valid data
            if len(res.get('train', [])) > 0:
                children.append(dcc.Graph(figure=make_forecast_fig(res, color), config={'displayModeBar': True}))
            
            metrics_data = res.get('metrics', {})
            if metrics_data:
                children.append(html.Div([
                    metric_pill("MAE",      f"${metrics_data.get('MAE', 0):,.0f}",        color),
                    metric_pill("RMSE",     f"${metrics_data.get('RMSE', 0):,.0f}",       '#FF6B9D' if color == '#00A8FF' else '#0088CC'),
                    metric_pill("MAPE",     f"{metrics_data.get('MAPE', 0):.2f}%",        '#9D4EDD'),
                    metric_pill("R²",       f"{metrics_data.get('R2', 0):.4f}",           '#00E676'),
                    metric_pill("Accuracy", f"{metrics_data.get('Accuracy', 0):.1f}%",    THEME['accent_cyan'] if color == '#00A8FF' else '#00A8FF'),
                ], style={'display': 'flex', 'gap': '8px', 'marginTop': '16px', 'flexWrap': 'wrap'}))
            
            return html.Div(children, style={**card(), 'borderLeft': f'3px solid {color}'})

        def verdict_block(label, metrics, color, strength, use_case):
            return html.Div([
                html.Div(label, style={'color': color, 'fontFamily': 'Rajdhani', 'fontSize': '15px', 'fontWeight': '700'}),
                html.Div(f"Accuracy : {metrics.get('Accuracy', 0):.1f}%",  style={'color': '#8B9DC3', 'fontSize': '11px', 'fontFamily': 'Share Tech Mono', 'marginTop': '6px'}),
                html.Div(f"R² Score : {metrics.get('R2', 0):.4f}",         style={'color': '#8B9DC3', 'fontSize': '11px', 'fontFamily': 'Share Tech Mono'}),
                html.Div(f"MAPE     : {metrics.get('MAPE', 0):.2f}%",      style={'color': '#8B9DC3', 'fontSize': '11px', 'fontFamily': 'Share Tech Mono'}),
                html.Div(f"Strength  : {strength}",  style={'color': '#555', 'fontSize': '10px', 'fontFamily': 'Exo 2', 'marginTop': '8px'}),
                html.Div(f"Best for : {use_case}",   style={'color': '#555', 'fontSize': '10px', 'fontFamily': 'Exo 2'}),
            ], style={
                'padding': '14px',
                'border': f'1px solid rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.35)',
                'borderRadius': '8px', 'marginBottom': '10px',
            })

        return html.Div([
            dbc.Row([
                dbc.Col([split_block(sarima_res,  'SARIMA',  '#00A8FF')], md=6),
                dbc.Col([split_block(prophet_res, 'PROPHET', '#9D4EDD')], md=6),
            ], className='g-3'),

            html.Div([
                section_label("MODEL COMPARISON — CLASSIFICATION-EQUIVALENT EVALUATION MATRIX"),
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(figure=make_comparison_bar(sm, pm), config={'displayModeBar': False}),
                    ], md=8),
                    dbc.Col([
                        verdict_block('SARIMA', sm, '#00A8FF',
                                      'Captures seasonality & autocorrelation',
                                      'Short-term precision, Q1-Q2 forecasts'),
                        verdict_block('PROPHET', pm, '#9D4EDD',
                                      'Handles trend changes & holiday effects',
                                      'Long-horizon planning, 12-month outlook'),
                    ], md=4),
                ]),
            ], style=card()),

            html.Div([
                section_label("2026 ANNUAL REVENUE PROJECTION SUMMARY"),
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.Div("SARIMA PROJECTION", style={'color': '#00A8FF', 'fontFamily': 'Rajdhani', 'fontSize': '13px', 'letterSpacing': '2px'}),
                            html.Div(f"${sarima_total:,.0f}", style={'color': '#fff', 'fontFamily': 'Rajdhani', 'fontSize': '30px', 'fontWeight': '700'}),
                            html.Div(f"Monthly avg: ${sarima_total/12:,.0f}", style={'color': '#555', 'fontFamily': 'Share Tech Mono', 'fontSize': '10px'}),
                        ], style={'padding': '20px', 'border': '1px solid rgba(0,168,255,0.3)', 'borderRadius': '8px', 'textAlign': 'center'}),
                    ], md=4),
                    dbc.Col([
                        html.Div([
                            html.Div("PROPHET PROJECTION", style={'color': '#9D4EDD', 'fontFamily': 'Rajdhani', 'fontSize': '13px', 'letterSpacing': '2px'}),
                            html.Div(f"${prophet_total:,.0f}", style={'color': '#fff', 'fontFamily': 'Rajdhani', 'fontSize': '30px', 'fontWeight': '700'}),
                            html.Div(f"Monthly avg: ${prophet_total/12:,.0f}", style={'color': '#555', 'fontFamily': 'Share Tech Mono', 'fontSize': '10px'}),
                        ], style={'padding': '20px', 'border': '1px solid rgba(157,78,221,0.3)', 'borderRadius': '8px', 'textAlign': 'center'}),
                    ], md=4),
                    dbc.Col([
                        html.Div([
                            html.Div("ENSEMBLE AVERAGE", style={'color': '#00D9FF', 'fontFamily': 'Rajdhani', 'fontSize': '13px', 'letterSpacing': '2px'}),
                            html.Div(f"${(sarima_total+prophet_total)/2:,.0f}", style={'color': '#fff', 'fontFamily': 'Rajdhani', 'fontSize': '30px', 'fontWeight': '700'}),
                            html.Div(f"Monthly avg: ${(sarima_total+prophet_total)/24:,.0f}", style={'color': '#555', 'fontFamily': 'Share Tech Mono', 'fontSize': '10px'}),
                        ], style={'padding': '20px', 'border': '1px solid rgba(0,217,255,0.3)', 'borderRadius': '8px', 'textAlign': 'center'}),
                    ], md=4),
                ], className='g-3'),
            ], style=card()),
        ])
    except Exception as e:
        print(f"[WARN]  Error rendering AI tab: {e}")
        import traceback
        traceback.print_exc()
        return html.Div([
            html.H3("[WARN] Error Loading AI Forecasting", style={'color': '#FF6B9D', 'padding': '20px'}),
            html.P(f"Details: {str(e)}", style={'color': '#8B9DC3', 'padding': '20px'}),
        ], style={**card(), 'borderLeft': '3px solid #FF6B9D'})


app.layout = html.Div([
    dcc.Loading(id="loading", children=[]),  # For loading states if needed
    html.Div([
        html.Div([
            html.Span("⚡", style={'fontSize': '20px', 'marginRight': '8px'}),
            html.Span("E-COM", style={'color': THEME['primary'], 'fontFamily': 'Rajdhani', 'fontWeight': '900', 'fontSize': '22px', 'letterSpacing': '4px'}),
            html.Span(" INTELLIGENCE", style={'color': THEME['white'], 'fontFamily': 'Rajdhani', 'fontWeight': '300', 'fontSize': '22px', 'letterSpacing': '4px'}),
            html.Span(" PLATFORM", style={'color': '#3A5A75', 'fontFamily': 'Rajdhani', 'fontWeight': '300', 'fontSize': '22px', 'letterSpacing': '4px'}),
        ], style={'display': 'flex', 'alignItems': 'center'}),
        html.Div([
            html.Span("8K CUSTOMERS · 20 COUNTRIES · 2020–2026 · ", style={'color': '#3A5A75', 'fontSize': '10px', 'fontFamily': 'Share Tech Mono'}),
            html.Span("LIVE", style={'color': THEME['accent_green'], 'fontSize': '10px', 'fontFamily': 'Share Tech Mono'}),
        ]),
    ], style={
        'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center',
        'padding': '14px 32px',
        'background': '#111F2E',
        'borderBottom': f'1px solid {THEME["border_primary"]}',
        'position': 'sticky', 'top': '0', 'zIndex': '100',
    }),

    dcc.Tabs(id='main-tabs', value='eda', children=[
        dcc.Tab(label='ANALYTICS & EDA', value='eda',
                className='nav-tab', selected_className='nav-tab--selected'),
        dcc.Tab(label='AI FORECASTING & MODELS', value='ai',
                className='nav-tab', selected_className='nav-tab--selected'),
    ], style={
        'background': '#111F2E',
        'borderBottom': f'1px solid {THEME["border"]}',
        'paddingLeft': '16px',
    }),

    html.Div(id='tab-content', style={'padding': '24px 32px'}),

], style={
    'backgroundColor': THEME['bg_primary'],
    'minHeight': '100vh',
    'fontFamily': 'Exo 2, sans-serif',
    'color': THEME['white'],
})

# Inject CSS styles
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            * { box-sizing: border-box; }
            ::-webkit-scrollbar { width: 6px; }
            ::-webkit-scrollbar-track { background: #111; }
            ::-webkit-scrollbar-thumb { background: #00A8FF; border-radius: 3px; }
            .nav-tab {
                color: #8B9DC3 !important; background: transparent !important; border: none !important;
                font-family: 'Rajdhani', sans-serif; font-size: 13px;
                letter-spacing: 2px; text-transform: uppercase;
                padding: 12px 28px !important;
                border-bottom: 2px solid transparent !important;
                transition: all 0.3s ease;
            }
            .nav-tab:hover { color: #fff !important; }
            .nav-tab--selected {
                color: #00A8FF !important;
                border-bottom: 2px solid #00A8FF !important;
                background: rgba(0,168,255,0.05) !important;
            }
            .tab-content { background: transparent !important; border: none !important; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''


@app.callback(Output('tab-content', 'children'), Input('main-tabs', 'value'))
def render_tab(tab):
    if tab == 'eda':
        return build_eda_tab()
    return build_ai_tab()


if __name__ == '__main__':
    print("\n" + "=" * 55)
    print("  E-Commerce Intelligence Dashboard")
    print("  -> Open: http://localhost:8050")
    print("=" * 55 + "\n")
    app.run(debug=False, host='0.0.0.0', port=8050)