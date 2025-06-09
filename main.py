import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf

# Створення тестових даних (замість реальних API)
def generate_stock_data(symbol, days=365):
    """Генерує тестові дані для акцій"""
    dates = pd.date_range(start=datetime.now() - timedelta(days=days), 
                         end=datetime.now(), freq='D')
    
    # Створюємо реалістичні дані з трендом та волатильністю
    np.random.seed(42 if symbol == 'AAPL' else 123)
    base_price = {'AAPL': 150, 'GOOGL': 2500, 'MSFT': 300, 'TSLA': 800}.get(symbol, 100)
    
    prices = []
    current_price = base_price
    
    for i in range(len(dates)):
        # Додаємо трендову складову та випадкові коливання
        trend = 0.0002 * i  # Загальний ростовий тренд
        volatility = np.random.normal(0, 0.02)
        current_price = current_price * (1 + trend + volatility)
        prices.append(current_price)
    
    return pd.DataFrame({
        'Date': dates,
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, len(dates)),
        'High': [p * (1 + np.random.uniform(0, 0.05)) for p in prices],
        'Low': [p * (1 - np.random.uniform(0, 0.05)) for p in prices]
    })

# Отримання реальних даних через API)
def fetch_stock_data(symbol, days=365):
    """Завантажує реальні історичні дані з yfinance"""
    end = datetime.now()
    start = end - timedelta(days=days)
    df = yf.download(symbol, start=start, end=end)
    
    if df.empty:
        raise ValueError(f"Дані для {symbol} не знайдені.")
    
    df = df.reset_index()
    df = df.rename(columns={
        'Date': 'Date',
        'Open': 'Open',
        'High': 'High',
        'Low': 'Low',
        'Close': 'Close',
        'Volume': 'Volume'
    })
    return df

# Ініціалізація додатку
app = dash.Dash(__name__)

# Початкові дані
stocks = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
stock_data = {stock: generate_stock_data(stock) for stock in stocks}

# Стилі
colors = {
    'background': '#f8f9fa',
    'text': '#2c3e50',
    'primary': '#3498db',
    'secondary': '#2ecc71',
    'accent': '#e74c3c'
}

# Макет додатку
app.layout = html.Div([
    html.Div([
        html.H1("📈 Фінансовий Дашборд", 
                style={'textAlign': 'center', 'color': colors['text'], 'marginBottom': '30px'}),
        
        # Панель управління
        html.Div([
            html.Div([
                html.Label("Виберіть акцію:", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                dcc.Dropdown(
                    id='stock-dropdown',
                    options=[{'label': f'{stock} Corporation', 'value': stock} for stock in stocks],
                    value='AAPL',
                    style={'marginBottom': '20px'}
                )
            ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '5%'}),
            
            html.Div([
                html.Label("Період часу:", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                dcc.DatePickerRange(
                    id='date-range',
                    start_date=(datetime.now() - timedelta(days=90)).date(),
                    end_date=datetime.now().date(),
                    display_format='DD/MM/YYYY',
                    style={'marginBottom': '20px'}
                )
            ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '5%'}),
            
            html.Div([
                html.Label("Тип графіку:", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                dcc.RadioItems(
                    id='chart-type',
                    options=[
                        {'label': ' Лінійний', 'value': 'line'},
                        {'label': ' Свічковий', 'value': 'candlestick'}
                    ],
                    value='line',
                    style={'marginBottom': '20px'}
                )
            ], style={'width': '30%', 'display': 'inline-block'}),
        ], style={'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '10px', 
                 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'marginBottom': '20px'}),
        
        # Основний графік
        html.Div([
            dcc.Graph(id='main-price-chart')
        ], style={'marginBottom': '20px'}),
        
        # Додаткові візуалізації
        html.Div([
            html.Div([
                dcc.Graph(id='volume-chart')
            ], style={'width': '48%', 'display': 'inline-block'}),
            
            html.Div([
                dcc.Graph(id='returns-histogram')
            ], style={'width': '48%', 'display': 'inline-block', 'marginLeft': '4%'})
        ], style={'marginBottom': '20px'}),
        
        # Статистика
        html.Div([
            html.H3("📊 Статистика", style={'color': colors['text'], 'marginBottom': '15px'}),
            html.Div(id='statistics-table')
        ], style={'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '10px',
                 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'marginBottom': '20px'}),
        
        # Кнопки для експорту
        html.Div([
            html.Button('📥 Завантажити дані (CSV)', id='download-btn', 
                       style={'backgroundColor': colors['primary'], 'color': 'white',
                             'border': 'none', 'padding': '10px 20px', 'borderRadius': '5px',
                             'marginRight': '10px', 'cursor': 'pointer'}),
            html.Button('🔄 Оновити дані', id='refresh-btn',
                       style={'backgroundColor': colors['secondary'], 'color': 'white',
                             'border': 'none', 'padding': '10px 20px', 'borderRadius': '5px',
                             'cursor': 'pointer'}),
            dcc.Download(id="download-dataframe-csv")
        ], style={'textAlign': 'center', 'marginTop': '20px'})
        
    ], style={'maxWidth': '1200px', 'margin': '0 auto', 'padding': '20px'})
], style={'backgroundColor': colors['background'], 'minHeight': '100vh'})

# Callback для основного графіку
@app.callback(
    Output('main-price-chart', 'figure'),
    [Input('stock-dropdown', 'value'),
     Input('date-range', 'start_date'),
     Input('date-range', 'end_date'),
     Input('chart-type', 'value')]
)
def update_main_chart(selected_stock, start_date, end_date, chart_type):
    df = stock_data[selected_stock].copy()
    
    # Фільтрація за датами
    df_filtered = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    
    if chart_type == 'line':
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_filtered['Date'],
            y=df_filtered['Close'],
            mode='lines',
            name=f'{selected_stock} Ціна',
            line=dict(color=colors['primary'], width=2)
        ))
    else:  # candlestick
        fig = go.Figure(data=go.Candlestick(
            x=df_filtered['Date'],
            open=df_filtered['Close'] * 0.99,  # Апроксимація
            high=df_filtered['High'],
            low=df_filtered['Low'],
            close=df_filtered['Close'],
            name=f'{selected_stock} Свічки'
        ))
    
    fig.update_layout(
        title=f'Динаміка ціни акцій {selected_stock}',
        xaxis_title='Дата',
        yaxis_title='Ціна ($)',
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig

# Callback для графіку об'єму торгів
@app.callback(
    Output('volume-chart', 'figure'),
    [Input('stock-dropdown', 'value'),
     Input('date-range', 'start_date'),
     Input('date-range', 'end_date')]
)
def update_volume_chart(selected_stock, start_date, end_date):
    df = stock_data[selected_stock].copy()
    df_filtered = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_filtered['Date'],
        y=df_filtered['Volume'],
        name='Об\'єм торгів',
        marker_color=colors['secondary']
    ))
    
    fig.update_layout(
        title=f'Об\'єм торгів {selected_stock}',
        xaxis_title='Дата',
        yaxis_title='Об\'єм',
        template='plotly_white'
    )
    
    return fig

# Callback для гістограми доходності
@app.callback(
    Output('returns-histogram', 'figure'),
    [Input('stock-dropdown', 'value'),
     Input('date-range', 'start_date'),
     Input('date-range', 'end_date')]
)
def update_returns_histogram(selected_stock, start_date, end_date):
    df = stock_data[selected_stock].copy()
    df_filtered = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    
    # Розрахунок денної доходності
    df_filtered['Returns'] = df_filtered['Close'].pct_change() * 100
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df_filtered['Returns'].dropna(),
        nbinsx=30,
        name='Денна доходність',
        marker_color=colors['accent']
    ))
    
    fig.update_layout(
        title=f'Розподіл денної доходності {selected_stock}',
        xaxis_title='Доходність (%)',
        yaxis_title='Частота',
        template='plotly_white'
    )
    
    return fig

# Callback для статистики
@app.callback(
    Output('statistics-table', 'children'),
    [Input('stock-dropdown', 'value'),
     Input('date-range', 'start_date'),
     Input('date-range', 'end_date')]
)
def update_statistics(selected_stock, start_date, end_date):
    df = stock_data[selected_stock].copy()
    df_filtered = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    
    # Розрахунок статистик
    current_price = df_filtered['Close'].iloc[-1]
    price_change = df_filtered['Close'].iloc[-1] - df_filtered['Close'].iloc[0]
    price_change_pct = (price_change / df_filtered['Close'].iloc[0]) * 100
    
    df_filtered['Returns'] = df_filtered['Close'].pct_change() * 100
    volatility = df_filtered['Returns'].std()
    max_price = df_filtered['Close'].max()
    min_price = df_filtered['Close'].min()
    avg_volume = df_filtered['Volume'].mean()
    
    stats = [
        html.Div([
            html.Div([
                html.H4(f"${current_price:.2f}", style={'margin': '0', 'color': colors['primary']}),
                html.P("Поточна ціна", style={'margin': '0', 'fontSize': '14px'})
            ], style={'textAlign': 'center', 'padding': '15px', 'backgroundColor': '#ecf0f1', 
                     'borderRadius': '8px', 'width': '18%', 'display': 'inline-block', 'margin': '1%'}),
            
            html.Div([
                html.H4(f"{price_change_pct:+.2f}%", 
                       style={'margin': '0', 'color': colors['secondary'] if price_change_pct >= 0 else colors['accent']}),
                html.P("Зміна за період", style={'margin': '0', 'fontSize': '14px'})
            ], style={'textAlign': 'center', 'padding': '15px', 'backgroundColor': '#ecf0f1', 
                     'borderRadius': '8px', 'width': '18%', 'display': 'inline-block', 'margin': '1%'}),
            
            html.Div([
                html.H4(f"{volatility:.2f}%", style={'margin': '0', 'color': colors['text']}),
                html.P("Волатильність", style={'margin': '0', 'fontSize': '14px'})
            ], style={'textAlign': 'center', 'padding': '15px', 'backgroundColor': '#ecf0f1', 
                     'borderRadius': '8px', 'width': '18%', 'display': 'inline-block', 'margin': '1%'}),
            
            html.Div([
                html.H4(f"${max_price:.2f}", style={'margin': '0', 'color': colors['secondary']}),
                html.P("Максимум", style={'margin': '0', 'fontSize': '14px'})
            ], style={'textAlign': 'center', 'padding': '15px', 'backgroundColor': '#ecf0f1', 
                     'borderRadius': '8px', 'width': '18%', 'display': 'inline-block', 'margin': '1%'}),
            
            html.Div([
                html.H4(f"{avg_volume:,.0f}", style={'margin': '0', 'color': colors['text']}),
                html.P("Сер. об'єм", style={'margin': '0', 'fontSize': '14px'})
            ], style={'textAlign': 'center', 'padding': '15px', 'backgroundColor': '#ecf0f1', 
                     'borderRadius': '8px', 'width': '18%', 'display': 'inline-block', 'margin': '1%'})
        ])
    ]
    
    return stats

# Callback для завантаження даних
@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("download-btn", "n_clicks"),
    [State('stock-dropdown', 'value'),
     State('date-range', 'start_date'),
     State('date-range', 'end_date')],
    prevent_initial_call=True
)
def download_data(n_clicks, selected_stock, start_date, end_date):
    if n_clicks:
        df = stock_data[selected_stock].copy()
        df_filtered = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
        return dcc.send_data_frame(df_filtered.to_csv, f"{selected_stock}_data.csv")

# Callback для оновлення даних
@app.callback(
    Output('stock-dropdown', 'options'),
    Input('refresh-btn', 'n_clicks'),
    prevent_initial_call=True
)
def refresh_data(n_clicks):
    if n_clicks:
        # Генеруємо нові дані
        global stock_data
        stock_data = {stock: generate_stock_data(stock) for stock in stocks}
    
    return [{'label': f'{stock} Corporation', 'value': stock} for stock in stocks]

if __name__ == '__main__':
    app.run(debug=True, port=8050)