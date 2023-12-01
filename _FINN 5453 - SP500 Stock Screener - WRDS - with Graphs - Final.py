import wrds
import pandas as pd
import numpy as np
import dash
from dash import dash_table
from dash import dcc, html, Input, Output
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import webbrowser
import threading
import time
import os
import logging
from concurrent.futures import ThreadPoolExecutor

def fetch_sp500_tickers():
    tables = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    df = tables[0]
    tickers = df['Symbol'].tolist()
    tickers = [ticker.replace('.', '-') for ticker in tickers]
    return tickers

sp500_tickers = fetch_sp500_tickers()
fetched_data = None

# Format the tickers list for SQL query
formatted_tickers = "'" + "', '".join(sp500_tickers) + "'"

# Save the formatted tickers to a CSV file
with open('formatted_tickers.csv', 'w') as file:
    file.write(formatted_tickers)

# Initialize the app
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Establish a connection
# Define a function to read credentials from a file
def read_credentials(filename):
    credentials = {}
    with open(filename, 'r') as f:
        for line in f:
            if "=" in line:
                key, value = line.strip().split('=', 1)
                credentials[key] = value
    return credentials

# Use the function to get credentials
credentials = read_credentials('credentials.txt')
username = credentials.get('username')
password = credentials.get('password')

db = wrds.Connection(wrds_username=username, wrds_password=password)  # Using provided credentials

# SQL query to fetch SP500 stocks from CRSP
query_sp500_stocks = f"""
SELECT ticker, dlyclose AS price, dlycaldt AS date, primaryexch AS exchange, permno
FROM crsp.wrds_dsfv2_query
WHERE ticker IN ({formatted_tickers}) AND dlycaldt >= '2020-01-01' AND dlycaldt <= '2023-11-01'
ORDER BY ticker, dlycaldt;
"""

# SQL query to fetch net income and number of shares from CRSP/Compustat merged database
query_net_income = f"""
SELECT tic AS ticker, datadate AS date, ni AS net_income_in_millions, csho AS number_of_shares_in_millions
FROM comp_na_annual_all.funda
WHERE tic IN ({formatted_tickers}) AND datadate >= '2020-01-01' AND datadate <= '2023-11-01'
ORDER BY tic, datadate;
"""

# SQL query to fetch data from Compustat secd
query_compustat_daily = f"""
SELECT tic AS ticker, cshoc AS number_of_shares, eps AS current_eps, datadate AS date
FROM comp.secd
WHERE tic IN ({formatted_tickers}) AND datadate >= '2020-01-01' AND datadate <= '2022-12-31'
ORDER BY tic, datadate;
"""

# SQL query to fetch financial ratios from IBES
query_financial_ratios = f"""
SELECT permno, qdate AS date, be AS book_equity, bm AS book_market_ratio, evm AS enterprise_value_multiple, pe_exi AS current_pe_ratio, ps AS price_sales_ratio,
      pcf AS price_cash_flow_ratio, npm AS net_profit_margin, opmbd AS operating_profit_margin, gpm AS gross_profit_margin, roa AS return_on_assets,
      roe AS return_on_equity, debt_ebitda AS debt_ebitda_ratio, cash_debt AS cash_debt_ratio, debt_assets AS debt_assets_ratio, de_ratio AS debt_equity_ratio,
      quick_ratio, curr_ratio AS current_ratio, mktcap AS mktcap_in_millions, ptb AS price_book_ratio, divyield AS dividend_yield_percentage,
      PEG_trailing AS peg_ratio, ticker, ret_crsp AS returns, fcf_ocf AS free_cash_flow_operating_cash_flow_ratio
FROM wrdsapps_finratio_ibes.firm_ratio_ibes
WHERE ticker IN ({formatted_tickers}) AND qdate >= '2022-09-01' AND qdate <= '2022-12-31'
ORDER BY ticker, qdate;
"""

# SQL query to fetch returns and net income from Compustat funda
query_revenue = f"""
SELECT datadate AS date, tic AS ticker, ni AS net_income_in_millions, revt AS total_revenue
FROM comp.funda
WHERE tic IN ({formatted_tickers}) AND datadate >= '2020-01-01' AND datadate <= '2022-12-31'
ORDER BY tic, datadate;
"""

# SQL query to fetch S&P 500 monthly returns from CRSP
# query_sp500_returns = """
# SELECT date, vwretd as market_return
# FROM crsp.msi
# WHERE date >= '2020-01-01' AND date <= '2022-12-31'
# ORDER BY date;
# """

# SQL query to fetch Beta from CRSP
query_beta1 = """
SELECT date, betav AS beta1, permno
FROM crsp_a_indexes.dport6
WHERE date >= '2022-12-01' AND date <= '2022-12-31'
ORDER BY permno, date;
"""

# SQL query to fetch Beta from CRSP
query_beta2 = """
SELECT date, betav AS beta2, permno
FROM crsp_a_indexes.dport8
WHERE date >= '2022-12-01' AND date <= '2022-12-31'
ORDER BY permno, date;
"""

# Define functions for each query
def fetch_sp500_stocks():
    return db.raw_sql(query_sp500_stocks)

def fetch_net_income():
    return db.raw_sql(query_net_income)

def fetch_compustat_daily():
    return db.raw_sql(query_compustat_daily)

def fetch_financial_ratios():
    return db.raw_sql(query_financial_ratios)

def fetch_revenue():
    return db.raw_sql(query_revenue)

# def fetch_sp500_returns():
    # return db.raw_sql(query_sp500_returns)

def fetch_beta1():
    return db.raw_sql(query_beta1)

def fetch_beta2():
    return db.raw_sql(query_beta2)

logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Funciton that preprocesses the data by dropping all columns that are entirely NA
def preprocess_dataframe(df):
    # Drop columns that are entirely NA
    df = df.dropna(axis=1, how='all')
    
    # Optionally, fill NA values in other columns if needed
    # df = df.fillna(0)  # Replace 0 with an appropriate fill value
    
    return df

# Use ThreadPoolExecutor to run tasks concurrently
def fetch_all_data():
    with ThreadPoolExecutor(max_workers=5) as executor:
        sp500_stocks = executor.submit(fetch_sp500_stocks)
        net_income = executor.submit(fetch_net_income)
        compustat_daily = executor.submit(fetch_compustat_daily)
        future_ratios = executor.submit(fetch_financial_ratios)
        revenue = executor.submit(fetch_revenue)
        # future_sp500 = executor.submit(fetch_sp500_returns)
        future_beta1 = executor.submit(fetch_beta1)
        future_beta2 = executor.submit(fetch_beta2)

        try:
            data_sp500_stocks = sp500_stocks.result()
            data_net_income = net_income.result()
            data_compustat_daily = compustat_daily.result()
            data_financial_ratios = future_ratios.result()
            data_revenue = revenue.result()
            # data_sp500_returns = future_sp500.result()
            data_beta1 = future_beta1.result()
            data_beta2 = future_beta2.result()
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
        
        # Add NYSE and NASDAQ exchange codes
        exchange_codes = {
            'N': 'NYSE',
            'Q': 'NASDAQ'
        }
        data_sp500_stocks['exchange'] = data_sp500_stocks['exchange'].map(exchange_codes)
        
        # Apply preprocessing to each DataFrame
        data_sp500_stocks = preprocess_dataframe(data_sp500_stocks)
        data_net_income = preprocess_dataframe(data_net_income)
        data_compustat_daily = preprocess_dataframe(data_compustat_daily)
        data_financial_ratios = preprocess_dataframe(data_financial_ratios)
        data_revenue = preprocess_dataframe(data_revenue)
        # data_sp500_returns = preprocess_dataframe(data_sp500_returns)
        data_beta1 = preprocess_dataframe(data_beta1)
        data_beta2 = preprocess_dataframe(data_beta2)
                
        # Create csv file for each database
        data_sp500_stocks.to_csv('data_sp500_stocks.csv', index=False)
        data_net_income.to_csv('data_net_income.csv', index=False)
        data_compustat_daily.to_csv('data_compustat_daily.csv', index=False)
        data_financial_ratios.to_csv('data_financial_ratios.csv', index=False)
        data_revenue.to_csv('data_revenue.csv', index=False)
        # data_sp500_returns.to_csv('data_sp500_returns.csv', index=False)
        data_beta1.to_csv('data_beta1.csv', index=False)
        data_beta2.to_csv('data_beta2.csv', index=False)

        # Merge the beta1 data with financial ratios on the 'permno' column
        merged_data = pd.merge(data_beta1, data_financial_ratios, on=['permno'], how='right')
        merged_data.to_csv('merged_data.csv', index=False)
        
        # Now merge the above with beta2 data on the 'permno' column
        merged2_data = pd.merge(merged_data, data_beta2, on=['permno'], how='left')
        merged2_data.to_csv('merged2_data.csv', index=False)
        
        # Merge 'date', 'date_x', and 'date_y' columns
        merged2_data['combined_date'] = merged2_data['date'].fillna(merged2_data['date_x'])
        # merged2_data['combined_date'] = merged2_data['date'].fillna(merged2_data['date_y'])
        # Drop the original 'date' and 'date_x', and 'date_y' columns
        merged2_data.drop(['date', 'date_x'], axis=1, inplace=True)
        merged2_data.to_csv('merged2_data.csv', index=False)
        # Rename 'combined_date' column to 'date'
        merged2_data.rename(columns={'combined_date': 'date'}, inplace=True)
        merged2_data.to_csv('merged2_data2.csv', index=False)
        
        # Merge the above data with compustat daily with the 'ticker' and 'date' columns
        merged3_data = pd.merge(merged2_data, data_compustat_daily, on= ['ticker', 'date'], how='left')
        merged3_data.to_csv('merged3_data.csv', index=False)
        
        # Merge the above with sp500 returns data on the 'date' column
        # merged4_data = pd.merge(merged3_data, data_sp500_returns, on=['date'], how='left')
        # merged4_data.to_csv('merged4_data.csv', index=False)

        # Merge the above with sp500 stocks data on the 'ticker' and 'date' column
        merged4_data = pd.merge(merged3_data, data_sp500_stocks, on=['ticker', 'date'], how='left')
        merged4_data.to_csv('merged4_data.csv', index=False)

        # Group by 'ticker' and 'date' and fill NA values using ffill()
        data_revenue['total_revenue'] = data_revenue.groupby(['ticker', 'date'])['total_revenue'].ffill()
        # Convert 'date' column to datetime
        data_revenue['date'] = pd.to_datetime(data_revenue['date'])
        # Sort the data by 'date' and group by 'ticker' and select the last row
        data_revenue = data_revenue.sort_values('date').groupby('ticker').last().reset_index()
        data_revenue.to_csv('data_revenue2.csv', index=False)
        
        # Merge the above with revenue data on the 'ticker' column
        final_merged_data = pd.merge(merged4_data, data_revenue, on=['ticker'], how='left')
        final_merged_data.rename(columns={'date_x': 'date'}, inplace=True)
        final_merged_data.to_csv('final_merged_data.csv', index=False)
  
        # Merge 'beta1' and 'beta2' columns
        final_merged_data['combined_beta'] = final_merged_data['beta1'].fillna(final_merged_data['beta2'])
        # Drop the original 'beta1' and 'beta2' columns
        final_merged_data.drop(['beta1', 'beta2'], axis=1, inplace=True)
        final_merged_data.to_csv('final_merged_data.csv', index=False)
        
        # Remove duplicates based on specific columns
        final_data = final_merged_data.drop_duplicates(subset=['ticker', 'date'])
        # Reset the index after dropping duplicates
        final_data = final_data.reset_index(drop=True)

        final_data['min_price'] = final_data['price']  # Create a new column 'min_price' with the same data as 'price'
        final_data['max_price'] = final_data['price']  # Similarly, create 'max_price'
        
        final_data['include_nyse'] = final_data['exchange'] == 'NYSE'  # Create a new column 'NYSE' with boolean values
        final_data['include_nasdaq'] = final_data['exchange'] == 'NASDAQ'  # Similarly, create 'NASDAQ'
        
        # Forward-fill the number_of_shares and net_income columns
        data_net_income['number_of_shares_in_millions'] = data_net_income.groupby('ticker')['number_of_shares_in_millions'].ffill()
        data_net_income['net_income_in_millions'] = data_net_income.groupby('ticker')['net_income_in_millions'].ffill()
        
        # Convert market cap to acutal $$
        final_data['market_capitalization'] = final_data['mktcap_in_millions'] * 1e6
        
        # Convert book equity to acutal $$
        final_data['book_equity'] = final_data['book_equity'] * 1e6
        
        # Convert net income to acutal $$
        final_data['net_income'] = final_data['net_income_in_millions'] * 1e6
        # Convert net income to acutal $$
        data_net_income['net_income'] = data_net_income['net_income_in_millions'] * 1e6
        
        # Convert number of shares to acutal total
        data_net_income['number_of_shares'] = data_net_income['number_of_shares_in_millions'] * 1e6
        
        # Convert total revenue to acutal $$
        final_data['total_revenue'] = final_data['total_revenue'] * 1e6
        
        # Convert dividend yield to a % value
        final_data['dividend_yield_percentage'] = final_data['dividend_yield_percentage'] * 100

        # Sort the final dataframe by ticker and date
        final_data = final_data.sort_values(by=['ticker', 'date'])
        data_sp500_stocks = data_sp500_stocks.sort_values(by=['ticker', 'date'])

        final_data.to_csv('final_data.csv', index=False)
        data_sp500_stocks.to_csv('data_sp500_stocks', index=False)

        # Calculate PEG ratio ratio where PEG ratio data is blank
        data_net_income['calculated_current_eps'] = data_net_income['net_income'] / data_net_income['number_of_shares']
        # Fill NA values using forward-fill or another appropriate method
        data_net_income['eps_growth_rate'] = data_net_income.groupby('ticker')['calculated_current_eps'].pct_change(periods=4, fill_method=None) * 100
        # Iterate through final_data and calculate PEG ratio where needed
        for index, row in final_data.iterrows():
            if pd.isna(row['peg_ratio']):
                # Fetch the corresponding eps_growth_rate from data_net_income
                eps_growth_rate = data_net_income.loc[data_net_income['ticker'] == row['ticker'], 'eps_growth_rate'].iloc[-1]
                if eps_growth_rate != 0:  # Avoid division by zero
                    calculated_peg_ratio = row['current_pe_ratio'] / eps_growth_rate
                    final_data.at[index, 'peg_ratio'] = calculated_peg_ratio
        
        final_data.to_csv('final_data.csv', index=False)
        data_net_income.to_csv('data_net_income.csv', index=False)
                
        # Calculate current P/E ratio where P/E ratio data is blank
        # Iterate through the DataFrame and add calculated P/E ratio where needed
        for index, row in final_data.iterrows():
            if pd.isna(row['current_pe_ratio']):
                # Calculate the current P/E ratio directly
                if row['current_eps'] != 0:  # Avoid division by zero
                    calculated_current_pe_ratio = row['price'] / row['current_eps']
                    # Update the current_pe_ratio in the DataFrame
                    final_data.at[index, 'current_pe_ratio'] = calculated_current_pe_ratio
                    
        final_data.to_csv('final_data.csv', index=False)

        # Calculate Earnings Yield
        # Handle cases where the P/E ratio might be zero or null to avoid division by zero errors
        final_data['earnings_yield'] = (1 / final_data['current_pe_ratio'].replace(0, np.nan)) * 100

        # Calculate Free Cash Flow
        # Calculate Operating Cash Flow
        final_data['operating_cash_flow'] = final_data['market_capitalization'] / final_data['price_cash_flow_ratio']
        # Calculate Free Cash Flow
        final_data['free_cash_flow'] = final_data['operating_cash_flow'] * final_data['free_cash_flow_operating_cash_flow_ratio']
        
        # Calculate Sharpe Ratio
        # Fill NA values before calculating daily returns using ffill() and sort data
        data_sp500_stocks['price'] = data_sp500_stocks.groupby('ticker')['price'].ffill()
        data_sp500_stocks['date'] = pd.to_datetime(data_sp500_stocks['date'])
        data_sp500_stocks.sort_values(by=['ticker', 'date'], inplace=True)
        data_sp500_stocks.to_csv('data_sp500_stocks1.csv', index=False)
        # Calculate daily returns without filling NA values
        data_sp500_stocks['daily_return'] = data_sp500_stocks.groupby('ticker')['price'].pct_change(fill_method=None)
        data_sp500_stocks.to_csv('data_sp500_stocks2.csv', index=False)
        # Calculate mean return and standard deviation of daily returns for each stock
        sharpe_stats = data_sp500_stocks.groupby('ticker')['daily_return'].agg(['mean', 'std']).reset_index()
        # Assuming an annual risk-free rate of 3.0%, or 0.0081% per day
        risk_free_rate = 0.000081
        # Calculate the Sharpe Ratio for each stock
        sharpe_stats['sharpe_ratio'] = (sharpe_stats['mean'] - risk_free_rate) / sharpe_stats['std']
        # Annualize the Sharpe Ratio
        sharpe_stats['sharpe_ratio'] = sharpe_stats['sharpe_ratio'] * np.sqrt(252)
        sharpe_stats.to_csv('sharpe_stats.csv', index=False)
        # Merge the Sharpe Ratio data into the final_data DataFrame
        final_data = pd.merge(final_data, sharpe_stats[['ticker', 'sharpe_ratio']], on='ticker', how='left')

        # Calculate Alpha
        # Assuming an annual risk-free rate of 3.0%, or 0.0081% per day
        risk_free_rate = 0.000081
        # Assuming an annual market return of 8.0%, or 0.021087% per day
        expected_market_return = 0.00021087
        # Calculate expected return using CAPM
        final_data['expected_return'] = risk_free_rate + final_data['combined_beta'] * (expected_market_return - risk_free_rate)
        # Calculate alpha
        final_data['alpha'] = final_data['returns'] - final_data['expected_return']
        # Annualize alpha
        final_data['alpha'] = final_data['alpha'] * np.sqrt(252)

        # Reorder the columns in the final data
        columns_order = ['ticker', 'price'] + [col for col in final_data.columns if col not in ['ticker', 'price']]
        final_data = final_data[columns_order]
        
        # Save to a CSV file
        final_data.to_csv('data_final_data.csv', index=False)

        # Return the final processed data
        return final_data
    
# Define app layout
def create_field(label, id, value, type='number'):
    if id in ['include_nyse', 'include_nasdaq']:
        # Setting the default value for both 'include_nyse' and 'include_nasdaq' to ['on']
        default_value = ['on']
        return html.Div([
            html.Label(label, style={'color': 'white'}),
            dcc.Checklist(
                id=id,
                options=[{'label': ' ', 'value': 'on'}],
                value=default_value,
                inline=True,
                style={'display': 'inline-block', 'margin-left': '10px'}
            )
        ], style={'width': '48%', 'display': 'inline-block'})
    else:
        return html.Div([
            html.Label(label, style={'color': 'white'}),
            dcc.Input(id=id, value=value, type=type, style={'color': 'black', 'backgroundColor': 'white'}),
            dcc.Checklist(
                id=f'check-{id}',
                options=[{'label': '', 'value': 'on'}],
                value=[],
                inline=True,
                style={'display': 'inline-block', 'margin-left': '10px'}
            )
        ], style={'width': '48%', 'display': 'inline-block'})

fields = [
    ('Min Price:', 'minPrice', 100),
    ('Max Price:', 'maxPrice', 200),
    ('Max P/E Ratio:', 'trailingPE', 20),
    ('Max PEG Ratio', 'pegRatio', 1),
    ('Min EPS:', 'trailingEps', 2),
    ('Min Dividend Yield:', 'dividendYield', .025), # Use decimal format
    ('Min P/B Ratio:', 'priceToBook', 1),
    ('Min Return on Equity:', 'returnOnEquity', 0.15), # Use decimal format
    ('Min Current Ratio:', 'currentRatio', 1.2),
    # ('Min Revenue Growth:', 'revenueGrowth', 0.05), # Use decimal format
    ('Min Free Cash Flow:', 'freeCashflow', 0.025),
    ('Min Operating Margin:', 'operating_margin', 0.15),
    ('Max Price/Sales Ratio:', 'priceToSalesTrailing12Months', 1),
    ('Min Earnings Yield:', 'earningsYield', 0.10),
    ('Min Quick Ratio:', 'quickRatio', 1.5),
    ('Max Debt/Equity Ratio:', 'debt_equity_ratio', 2),
    ('Min Alpha:', 'alpha', 0.10),
    ('Min Beta', 'beta', 1),
    ('Min Sharpe Ratio', 'sharpe_ratio', 1),
    ('Include NYSE Stocks', 'include_nyse', None),  # Checkbox for NYSE
    ('Include NASDAQ Stocks', 'include_nasdaq', None),  # Checkbox for NASDAQ
    ('Min Market Cap', 'min_market_cap', 1e9),
]

# Define layout
app.layout = html.Div([
    html.H1('SP500 Stock Screener', style={'textAlign': 'center', 'color': 'white'}),  # Add this line for the label
    dcc.Location(id='url', refresh=False),
    html.Div([create_field(*field) for field in fields]),
    html.Button('Screen Stocks', id='screen-button', style={'color': 'black', 'backgroundColor': 'white'}),
    dcc.Loading(
        id="loading",
        type="circle",
        children=[
            html.Div(id='output-table', style={'color': 'white'}),
        ],
        style={'margin-top': '-300px'}  # Adjust this value as needed
    ),
    html.Div(id='price-history')  # Div to display the price history chart
], style={'backgroundColor': '#000000', 'textAlign': 'center'})

# Updated callback for filtering and displaying the table
@app.callback(
    [Output('output-table', 'children')],
    [Input('screen-button', 'n_clicks')],
    [State(field_id, 'value') for _, field_id, _ in fields if field_id not in ['include_nyse', 'include_nasdaq']] +
    [State('include_nyse', 'value'), State('include_nasdaq', 'value')]
)
def update_output(n_clicks, *args):
    if n_clicks is None:
        return [dash.no_update]
    n = len(fields)
    values = args[:n]
    checkboxes = args[n:]
    
    df = fetch_all_data()
    if df is None or df.empty:
        return ['No data to display']
    
    # Convert 'date' column to datetime and find the most recent date
    df['date'] = pd.to_datetime(df['date'])
    most_recent_date = df['date'].max()

    # Filter the DataFrame to only include rows with the most recent date
    df = df[df['date'] == most_recent_date]
    
    # Create a mapping from the field identifier to the actual DataFrame column name
    field_to_df_column = {
        'minPrice': 'price',
        'maxPrice': 'price',
        'trailingPE': 'current_pe_ratio',
        'pegRatio': 'peg_ratio',
        'trailingEps': 'current_eps',
        'dividendYield': 'dividend_yield_percentage',
        'priceToBook': 'price_book_ratio',
        'returnOnEquity': 'return_on_equity',
        'currentRatio': 'current_ratio',
        # 'revenueGrowth': 'returns',
        'freeCashflow': 'free_cash_flow',
        'operating_margin': 'operating_profit_margin',
        'priceToSalesTrailing12Months': 'price_sales_ratio',
        'earningsYield': 'earnings_yield',
        'quickRatio': 'quick_ratio',
        'debt_equity_ratio': 'debt_equity_ratio',
        'alpha': 'alpha',
        'beta': 'combined_beta',
        'sharpe_ratio': 'sharpe_ratio',
        'min_market_cap': 'market_capitalization',
    }
    
    # NYSE and NASDAQ checkbox handling
    include_nyse = 'on' in args[-2]  # Assuming 'include_nyse' is second last in the args list
    include_nasdaq = 'on' in args[-1]  # Assuming 'include_nasdaq' is last in the args list
    if include_nyse and not include_nasdaq:
        df = df[df['exchange'] == 'NYSE']
    elif include_nasdaq and not include_nyse:
        df = df[df['exchange'] == 'NASDAQ']
    elif not include_nyse and not include_nasdaq:
        df = pd.DataFrame()  # Empty DataFrame if none are selected
    
    # Loop through all the criteria, and apply the filters only if the checkbox is checked ('on')
    for value, (label, field_id, _), checkbox in zip(values, fields, checkboxes):
        if 'on' not in checkbox:  # Check if checkbox is checked
            continue  # Skip this criterion if not checked
            
        column = field_to_df_column[field_id]
        
        if 'Min' in label:
            df = df.loc[df[column] >= value].copy()
        elif 'Max' in label:
            df = df.loc[df[column] <= value].copy()
    
    # Round all numeric columns to 4 decimal places
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].round(4)
        
    # Select only the specified columns
    selected_columns = ['ticker', 'price', 'current_pe_ratio', 'peg_ratio', 'current_eps', 'dividend_yield_percentage', 'price_book_ratio', 'return_on_equity',
                        'current_ratio', 'free_cash_flow', 'operating_profit_margin', 'price_sales_ratio', 'earnings_yield', 'quick_ratio', 'debt_equity_ratio',
                        'alpha', 'combined_beta', 'sharpe_ratio', 'market_capitalization', 'exchange']
    df = df[selected_columns]
    
    # Create Dash DataTable with an ID
    table = dash_table.DataTable(
        id='stock-table',
        columns=[{"name": i, "id": i} for i in df.columns],
        data=df.to_dict('records'),
        style_cell={
            'backgroundColor': 'black',
            'color': 'white',
            'border': '1px solid gray',
            'textAlign': 'center',
            'padding': '10px',
            'font-size': '12px',
            'font-family': 'sans-serif'
        },
        style_header={
            'backgroundColor': 'black',
            'color': 'white',
            'font-size': '12px',
            'fontWeight': 'bold',
            'border': '1px solid grey',
            'textAlign': 'center'
        },
        style_data={
            'whiteSpace': 'normal',
            'height': 'auto',
            'lineHeight': '15px',
            'border': '1px solid grey'
        },
        style_table={
            'backgroundColor': 'black',
            'height': '350px',
            'overflowY': 'scroll'
        }
    )
    return [table]
    
# Callback for handling clicks and displaying price history
@app.callback(
    Output('price-history', 'children'),
    [Input('stock-table', 'active_cell')],
    [State('stock-table', 'data')]
)
def display_price_history(active_cell, table_data):
    if not active_cell:
        return "Click on a ticker to see its price history."

    clicked_row_index = active_cell['row']
    ticker = table_data[clicked_row_index]['ticker']

    # Fetch historical data for the selected ticker
    sp500_stocks = fetch_sp500_stocks()
    df_history = sp500_stocks[sp500_stocks['ticker'] == ticker]

    if df_history.empty:
        return f"No historical data available for ticker: {ticker}"

    # Generate the price history chart
    fig = go.Figure(data=[go.Scatter(x=df_history['date'], y=df_history['price'], mode='lines', name=ticker)])
    fig.update_layout(title=f'Price History of {ticker}', xaxis_title='Date', yaxis_title='Price')

    return dcc.Graph(figure=fig)

def open_browser():
    # Wait for the server to start before opening the browser
    time.sleep(1)  # Adjust the delay as needed
    webbrowser.open("http://127.0.0.1:8050")

if __name__ == '__main__':
    if not os.environ.get('DASH_APP_RUNNING'):
        # Start a thread that will open the web browser
        threading.Thread(target=open_browser).start()
        os.environ['DASH_APP_RUNNING'] = '1'

    # Start the Dash app
    app.run_server(debug=False)

# Close the connection
db.close()