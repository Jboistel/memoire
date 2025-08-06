import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import norm
from scipy.stats import lognorm
""" data_path = "/home/onyx/Documents/MEMOIRE/memoire/data/sales_train_evaluation.csv"
pkl_path = "/home/onyx/Documents/MEMOIRE/memoire/data/sales_train_evaluation.pkl" """
data_path = "/home/onyx/Documents/Mémoire/memoire/data/sales_train_evaluation.csv"
pkl_path = "/home/onyx/Documents/Mémoire/memoire/data/sales_train_evaluation.pkl"
cal_path = "/home/onyx/Documents/Mémoire/memoire/data/calendar.csv"

REVIEW_TIME = 7
LEAD_TIME = 2

alpha = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99]
def load_data(data_path: str) -> pd.DataFrame:
    """
    Load data from a CSV or pickle file into a pandas DataFrame.
    
    Parameters:
        data_path (str): Path to the data file, can be CSV or pickle.
    
    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    if data_path.endswith(".csv"):
        # Load the CSV file
        df = pd.read_csv(data_path)
        df.to_pickle(pkl_path)
    elif data_path.endswith(".pkl"):
        # Load the pickle file
        df = pd.read_pickle(pkl_path)
    else:
        raise ValueError("Unsupported file format. Please use .csv or .pkl")

    return df
def scraping_high_zeroes_products(df: pd.DataFrame, n: int) -> pd.DataFrame:
    # Count the number of zeroes in each row
    df['zero_count'] = (df == 0).sum(axis=1)
    
    # Sort the DataFrame by the number of zeroes in ascending order
    df_sorted = df.sort_values(by='zero_count')
    
    # Keep only the top 1000 rows with the least number of zeroes
    df_top_n = df_sorted.head(n)
    
    # Drop the 'zero_count' column as it is no longer needed
    df_top_n = df_top_n.drop(columns=['zero_count'])
    
    return df_top_n

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the DataFrame by removing unnecessary columns and converting
    the demand columns to numeric types.
    
    Parameters:
        df (pd.DataFrame): Original DataFrame.
    
    Returns:
        pd.DataFrame: Demand matrix with product 'id' as index and days as columns.
    """
    # Remove unnecessary columns
    df_demand = df.drop(columns=['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'])
    # Set 'id' as index
    demand_matrix = df_demand.set_index('id')

    # Load the calendar.csv file
    calendar = pd.read_csv(cal_path)

    # Get the date columns from the calendar dataframe
    date_columns = calendar['date']
    demand_matrix.columns = date_columns[0:demand_matrix.shape[1]]
    
    
    return demand_matrix

def analyse_data(df: pd.DataFrame) -> [pd.Series, pd.Series]:
    means = df.mean(axis=1)
    stds = df.std(axis=1)
    
    return means, stds

def FC_avg_n_days(demand_matrix: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Compute the forecast by averaging the last n days for each product.
    
    Parameters:
        demand_matrix (pd.DataFrame): Demand matrix.
    
    Returns:
        pd.DataFrame: Forecast matrix (rounded up).
    """
    forecast_matrix = demand_matrix.T.rolling(window=n).mean().shift(1+REVIEW_TIME+LEAD_TIME).T
    return forecast_matrix

def FC_med_n_days(demand_matrix: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Compute the forecast by averaging the last n days for each product.
    
    Parameters:
        demand_matrix (pd.DataFrame): Demand matrix.
    
    Returns:
        pd.DataFrame: Forecast matrix (rounded up).
    """
    forecast_matrix = demand_matrix.T.rolling(window=n).median().shift(1+REVIEW_TIME+LEAD_TIME).T
    return forecast_matrix

def FC_avg_n_same_days(demand_matrix: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Compute the forecast by averaging the same day of the week over the previous 4 weeks for each product.
    
    Parameters:
        demand_matrix (pd.DataFrame): Demand matrix.
    
    Returns:
        pd.DataFrame: Weekly forecast matrix (rounded up).
    """
    demand_np = demand_matrix.to_numpy()
    n_products, n_days = demand_np.shape
    weekly_forecast_np = np.full((n_products, n_days), np.nan)
    
    # For each day of the week (offset from 0 to 6)
    for offset in range(7):
        idx = np.arange(offset, n_days, 7)
        # Need at least 5 weeks to use the previous n weeks
        if len(idx) < n+1:
            continue
        # Create a sliding window view for the previous n weeks
        slices = np.stack([demand_np[:, idx[i: i + len(idx) - n]] for i in range(1,n)], axis=-1)
        rolling_mean = np.mean(slices, axis=-1)
        target_indices = idx[n:]  # Days to forecast
        weekly_forecast_np[:, target_indices] = rolling_mean
        
    weekly_forecast_matrix = pd.DataFrame(
        weekly_forecast_np,
        index=demand_matrix.index,
        columns=demand_matrix.columns
    )
    return weekly_forecast_matrix

def FC_med_n_same_days(demand_matrix: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Compute the forecast by taking the median of the same day of the week over the previous 4 weeks for each product.
    
    Parameters:
        demand_matrix (pd.DataFrame): Demand matrix.
    
    Returns:
        pd.DataFrame: Weekly forecast matrix (rounded up).
    """
    demand_np = demand_matrix.to_numpy()
    n_products, n_days = demand_np.shape
    weekly_forecast_np = np.full((n_products, n_days), np.nan)
    
    # For each day of the week (offset from 0 to 6)
    for offset in range(7):
        idx = np.arange(offset, n_days, 7)
        # Need at least n+1 weeks to use the previous n weeks
        if len(idx) < n+1:
            continue
        # Create a sliding window view for the previous n weeks
        slices = np.stack([demand_np[:, idx[i: i + len(idx) - n]] for i in range(1,n)], axis=-1)
        rolling_med = np.median(slices, axis=-1)
        target_indices = idx[n:]  # Days to forecast
        weekly_forecast_np[:, target_indices] = rolling_med
        
    weekly_forecast_matrix = pd.DataFrame(
        weekly_forecast_np,
        index=demand_matrix.index,
        columns=demand_matrix.columns
    )
    return weekly_forecast_matrix

def SS_cst(demand_matrix: pd.DataFrame, forecast_matrix: pd.DataFrame, safety_stock_value: int, R: int) -> pd.DataFrame:
    """
    Create a safety stock matrix with a constant integer value for each product and each day.
    
    Parameters:
        demand_matrix (pd.DataFrame): Demand matrix with product index and days as columns.
        safety_stock_value (int): The safety stock level to assign.
    
    Returns:
        pd.DataFrame: Safety stock matrix with the same shape as demand_matrix, filled with safety_stock_value.
    """
    safety_stock_matrix = pd.DataFrame(
        safety_stock_value,
        index=demand_matrix.index,
        columns=demand_matrix.columns
    )
    return safety_stock_matrix

def SS_on_forecast(demand_matrix: pd.DataFrame, forecast_matrix: pd.DataFrame, safety_stock_factor: int, R: int) -> pd.DataFrame:
    """
    Create a safety stock matrix constituted of the n+3 day of forecast times the safety_stock_factor.

    Parameters:
        demand_matrix (pd.DataFrame): Demand matrix with product index and days as columns.
        safety_stock_factor: The factor to apply to the forecasted demand.

    Returns:
        pd.DataFrame: Safety stock matrix with the same shape as demand_matrix, filled with safety_stock_value.
    """
    safety_stock_matrix = pd.DataFrame(
        safety_stock_factor * forecast_matrix.shift(-R, axis=1).values,
        index=demand_matrix.index,
        columns=demand_matrix.columns
    )
    return safety_stock_matrix
def up_to_level_optimal_restock(demand_matrix, forecast_matrix: pd.DataFrame, safety_stock_matrix: pd.DataFrame, X, min_fill_rate, R):
    """
    Find the optimal fixed restock level that reaches the minimal fill rate.
    
    Parameters:
        min_fill_rate (float): The minimum fill rate to achieve.
    
    Returns:
        int: The optimal fixed restock level.
    """
    demand = demand_matrix.to_numpy()
    up_to_level = np.ones(demand_matrix.shape)
    _, _, _, _, _, lost_sales = compute_stock(up_to_level, demand)
    fill_rates_per_product = compute_fill_rates_per_product(lost_sales, demand)

    while not (fill_rates_per_product > min_fill_rate).all():
        _, _, _, _, _, lost_sales = compute_stock(up_to_level, demand)
        fill_rates_per_product = compute_fill_rates_per_product(lost_sales, demand)
        goods = np.where(fill_rates_per_product <= min_fill_rate)
        up_to_level[goods] += 1
    
    for i in range(up_to_level.shape[1]):
        if i + R <= up_to_level.shape[1]:
            up_to_level[:, i] = np.sum(up_to_level[:, i:i+R], axis=1)
        else:
            up_to_level[:, i] = np.sum(up_to_level[:, :], axis=1)
    
    return pd.DataFrame(up_to_level, index=demand_matrix.index, columns=demand_matrix.columns)

def compute_fill_rates_per_product(lost_sales, demand):
    """Compute the fill rate for each product
    
    Parameters:
        lost_sales (ndarray): Numpy array of lost sales.
        demand (ndarray): Numpy array of demand.
        
    Returns:
        ndarray: Numpy array of fill rates.
    """
    return 1 - lost_sales.sum(axis=1) / demand.sum(axis=1)
def up_to_level_on_forecast(demand_matrix, forecast_matrix: pd.DataFrame, safety_stock_matrix: pd.DataFrame, X, min_fill_rate, R):
    """
    Compute the forecast by summing the demand of the next 3 days for each product.
    
    Parameters:
        demand_matrix (pd.DataFrame): Demand matrix.
    
    Returns:
        pd.DataFrame: Forecast matrix (rounded up).
    """
    up_to_level_matrix = forecast_matrix.T.shift(-R+1).rolling(window=R).sum().T
    up_to_level_matrix += safety_stock_matrix
    
    return np.ceil(up_to_level_matrix)

def up_to_level_fixed_X(demand_matrix, forecast_matrix: pd.DataFrame, safety_stock_matrix: pd.DataFrame, X, min_fill_rate, R):
    ones = np.ones(demand_matrix.shape)
    up_to_level = ones * X
    up_to_level_matrix = pd.DataFrame(up_to_level, index=demand_matrix.index, columns=demand_matrix.columns)
    
    up_to_level = up_to_level_matrix.to_numpy()

    for i in range(up_to_level.shape[1]):
        if i + R <= up_to_level.shape[1]:
            up_to_level[:, i] = np.sum(up_to_level[:, i:i+R], axis=1)
        else:
            up_to_level[:, i] = np.sum(up_to_level[:, :], axis=1)

    up_to_level_matrix = pd.DataFrame(up_to_level, index=demand_matrix.index, columns=demand_matrix.columns)

    return up_to_level_matrix
def up_to_level_normal_demand( #TODO: DATA LEAKAGE
    demand_matrix: pd.DataFrame,
    forecast_matrix,       # non utilisé mais conservé pour pipeline
    safety_stock_matrix,   # idem
    X,                     # idem
    min_fill_rate: float, #CSL
    window_size: int = 14  # taille de la fenêtre glissante
) -> pd.DataFrame:
    """
    Même principe que before, mais on ne regarde que les `window_size` derniers jours
    pour estimer mean et std avant le jour j.
    """
    # 1) Rolling stats (on prend j−window_size à j−1, puis shift pour ne pas prendre j)
    roll_means = demand_matrix.rolling(
        window=window_size, axis=1, min_periods=1
    ).mean().shift(1+REVIEW_TIME+LEAD_TIME, axis=1).fillna(0)
    roll_stds = demand_matrix.rolling(
        window=window_size, axis=1, min_periods=1
    ).std(ddof=0).shift(1+REVIEW_TIME+LEAD_TIME, axis=1).fillna(0)

    # 2) Quantile
    quantiles = pd.DataFrame(
        norm.ppf(min_fill_rate, loc=roll_means.values, scale=roll_stds.values),
        index=demand_matrix.index,
        columns=demand_matrix.columns
    ).clip(lower=0)

    # 3) Somme glissante sur R jours
    n_days = demand_matrix.shape[1]
    up_to = pd.DataFrame(0, index=demand_matrix.index, columns=demand_matrix.columns)
    for j in range(n_days):
        end = min(j + REVIEW_TIME + LEAD_TIME, n_days)
        up_to.iloc[:, j] = quantiles.iloc[:, j:end].sum(axis=1)

    return np.ceil(up_to)

def up_to_level_lognormal_demand(
    demand_matrix: pd.DataFrame,
    forecast_matrix,       # non utilisé mais conservé pour pipeline
    safety_stock_matrix,   # idem
    X,                     # idem
    min_fill_rate: float,
    R: int,
    window_size: int = 14  # taille de la fenêtre glissante
) -> pd.DataFrame:
    """
    Même principe que before, mais on ne regarde que les `window_size` derniers jours
    pour estimer mean et std avant le jour j, avec une distribution log-normale.
    """
    # 1) Rolling stats (on prend j−window_size à j−1, puis shift)
    roll_means = demand_matrix.rolling(window=window_size, axis=1, min_periods=1).mean().shift(1, axis=1).fillna(0)
    roll_stds = demand_matrix.rolling(window=window_size, axis=1, min_periods=1).std(ddof=0).shift(1, axis=1).fillna(0)
    
    # 2) Convertir roll_means et roll_stds vers les paramètres de la lognormale
    # On suppose que roll_means > 0 pour effectuer le logarithme
    sigma = np.sqrt(np.log(1 + (roll_stds / roll_means)**2))
    mu = np.log(roll_means) - 0.5 * sigma**2

    # 3) Calculer le quantile en utilisant la lognormale : scale = exp(mu), s = sigma 
    quantile_vals = lognorm.ppf(min_fill_rate, s=sigma, scale=np.exp(mu))
    quantiles = pd.DataFrame(quantile_vals, index=demand_matrix.index, columns=demand_matrix.columns).clip(lower=0)
    
    # 4) Somme glissante sur R jours
    n_days = demand_matrix.shape[1]
    up_to = pd.DataFrame(0, index=demand_matrix.index, columns=demand_matrix.columns)
    for j in range(n_days):
        end = min(j + R, n_days)
        up_to.iloc[:, j] = quantiles.iloc[:, j:end].sum(axis=1)
    
    return np.ceil(up_to)
def compute_stock(up_to_level, demand, order_rate=1):
    """
    Compute various stock management indicators:
      - inventory_begin: stock at the beginning of the day
      - inventory_end: stock at the end of the day
      - T1: stock arriving in 1 day
      - T2: stock arriving in 2 days
      - sales: actual sales
      - lost_sales: unmet demand (lost sales)
    
    Parameters:
        up_to_level (ndarray): Matrix of levels to reach.
        demand (ndarray): Demand matrix.
        order_rate (int): Interval of days between orders.
    
    Returns:
        tuple of pd.DataFrame: DataFrames for inventory_begin, inventory_end, T1, T2, sales, and lost_sales.
    """

    """ # Conversion vers arrays
    demand = demand_matrix.to_numpy()
    up_to_level = up_to_level_matrix.to_numpy()
    # Ensure no NaN values in the matrices """
    np.nan_to_num(up_to_level, copy=False)

    n_products, n_days = demand.shape

    # Initialisation des arrays
    inventory_begin = np.zeros_like(demand)
    inventory_end = np.zeros_like(demand)
    T1 = np.zeros_like(demand)
    T2 = np.zeros_like(demand)
    sales = np.zeros_like(demand)
    lost_sales = np.zeros_like(demand)

    # Initialisation jour 0
    inventory_begin[:, 0] = 5
    T1[:, 0] = 0
    T2[:, 0] = up_to_level[:, 0] - inventory_begin[:, 0]
    inventory_end[:, 0] = np.maximum(inventory_begin[:, 0] - demand[:, 0], 0)
    sales[:, 0] = inventory_begin[:, 0] - inventory_end[:, 0]
    lost_sales[:, 0] = np.maximum(demand[:, 0] - sales[:, 0], 0)

    # Boucle vectorisée jour par jour
    for day in range(1, n_days):
        T1[:, day] = T2[:, day - 1]
        inventory_begin[:, day] = inventory_end[:, day - 1] + T1[:, day - 1]
        inventory_end[:, day] = np.maximum(inventory_begin[:, day] - demand[:, day], 0)

        if(day%order_rate == 0):
            T2[:, day] = np.nan_to_num(np.maximum(up_to_level[:, day] - inventory_end[:, day] - T1[:, day], 0))
        sales[:, day] = inventory_begin[:, day] - inventory_end[:, day]
        lost_sales[:, day] = np.maximum(demand[:, day] - sales[:, day], 0)

    return inventory_begin, inventory_end, T1, T2, sales, lost_sales

#inventory_begin, inventory_end, _, _, _, lost_sales = compute_stock(up_to_level_matrix.to_numpy(), demand_matrix.to_numpy())
def compute_results(demand, lost_sales, inventory_begin, inventory_end, n_days):
    """
    Compute overall stock management metrics:
      - Fill rate
      - Average inventory
      - Ratio of days with perfect service (no lost sales)
    
    Parameters:
        demand_matrix (pd.DataFrame): Demand matrix.
        forecast_matrix (pd.DataFrame): Forecast matrix.
        up_to_level_matrix (pd.DataFrame): Matrix of levels to reach.
    
    Returns:
        dict: Dictionary containing the results.
    """

    fill_rate = 1 - lost_sales.sum().sum() / demand.sum().sum()
    avg_inventory = (inventory_begin.sum().sum() + inventory_end.sum().sum()) / 2 // n_days
    # Compute the average percentage of products containing lost sales per day
    products_with_lost_sales = (lost_sales > 0).sum(axis=0)/lost_sales.shape[0]
    average_percentage_lost_sales = products_with_lost_sales.mean()
    
    return fill_rate, avg_inventory, average_percentage_lost_sales

def compute_pareto(_forecast_func=None, _fc_factor=None,
                    _ss_func=SS_cst, _ss_factor=0,
                    _up_to_level_func=None, _X=None, _min_fill_rate=None,
                    _compute_stock_func=None, _order_rate=1):
    results = {}

    if type(_ss_factor) == list:
        for i in _ss_factor:
            fill_rate, avg_inventory, average_percentage_lost_sales = pipeline(
                forecast_func=_forecast_func, fc_factor=_fc_factor,
                ss_func=_ss_func, ss_factor=i,
                up_to_level_func=_up_to_level_func, X=_X, min_fill_rate=_min_fill_rate,
                compute_stock_func=_compute_stock_func, order_rate=_order_rate
            )
            results[i] = (fill_rate, avg_inventory, average_percentage_lost_sales)
    elif type(_X) == list:
        for i in _X:
            fill_rate, avg_inventory, average_percentage_lost_sales = pipeline(
                forecast_func=_forecast_func, fc_factor=_fc_factor,
                ss_func=_ss_func, ss_factor=_ss_factor,
                up_to_level_func=_up_to_level_func, X=i, min_fill_rate=_min_fill_rate,
                compute_stock_func=_compute_stock_func, order_rate=_order_rate
            )
            results[i] = (fill_rate, avg_inventory, average_percentage_lost_sales)
    elif type(_min_fill_rate) == list:
        for i in _min_fill_rate:
            fill_rate, avg_inventory, average_percentage_lost_sales = pipeline(
                forecast_func=_forecast_func, fc_factor=_fc_factor,
                ss_func=_ss_func, ss_factor=_ss_factor,
                up_to_level_func=_up_to_level_func, X=_X, min_fill_rate=i,
                compute_stock_func=_compute_stock_func, order_rate=_order_rate
            )
            results[i] = (fill_rate, avg_inventory, average_percentage_lost_sales)
    elif type(_order_rate) == list:
        for i in _order_rate:
            fill_rate, avg_inventory, average_percentage_lost_sales = pipeline(
                forecast_func=_forecast_func, fc_factor=_fc_factor,
                ss_func=_ss_func, ss_factor=_ss_factor,
                up_to_level_func=_up_to_level_func, X=_X, min_fill_rate=_min_fill_rate,
                compute_stock_func=_compute_stock_func, order_rate=i
            )
            results[i] = (fill_rate, avg_inventory, average_percentage_lost_sales)
    else : 
        raise ValueError("No variable to iterate over")
    
    return results
def compute_fill_rates_per_product(lost_sales, demand):
    """Compute the fill rate for each product
    
    Parameters:
        lost_sales (ndarray): Numpy array of lost sales.
        demand (ndarray): Numpy array of demand.
        
    Returns:
        ndarray: Numpy array of fill rates.
    """
    return 1 - lost_sales.sum(axis=1) / demand.sum(axis=1)
#KPIs forecasts
def mape(demand, forecast):
    mask = demand != 0
    return np.mean(np.abs((demand[mask] - forecast[mask]) / demand[mask])) * 100

def rmse(demand, forecast):
    return np.sqrt(np.mean((demand - forecast)**2))

def mae(demand, forecast):
    return np.mean(np.abs(demand - forecast))

def bias(demand, forecast):
    return np.mean(forecast - demand)

def smape(demand, forecast):
    denominator = (np.abs(demand) + np.abs(forecast)) / 2
    mask = denominator != 0
    return np.mean(np.abs(demand[mask] - forecast[mask]) / denominator[mask]) * 100
def plot_stocks(demand_matrix, forecast_matrix, lost_sales, inventory_begin, inventory_end):
    
    # Transform in pd.DataFrame
    lost_sales_matrix = pd.DataFrame(lost_sales, index=demand_matrix.index, columns=demand_matrix.columns)
    inventory_begin_matrix = pd.DataFrame(inventory_begin, index=demand_matrix.index, columns=demand_matrix.columns)
    inventory_end_matrix = pd.DataFrame(inventory_end, index=demand_matrix.index, columns=demand_matrix.columns)

    # Scrap the first 7 days for the plot  
    demand_matrix = demand_matrix.iloc[:, 7:]
    forecast_matrix = forecast_matrix.iloc[:, 7:]
    lost_sales_matrix = lost_sales_matrix.iloc[:, 7:]
    inventory_begin_matrix = inventory_begin_matrix.iloc[:, 7:]
    inventory_end_matrix = inventory_end_matrix.iloc[:, 7:]
    
    # Agrégation
    total_demand = demand_matrix.sum(axis=0)
    total_forecast = forecast_matrix.sum(axis=0)
    total_inventory = (inventory_begin_matrix.sum(axis=0) + inventory_end_matrix.sum(axis=0)) / 2
    total_lost_sales = lost_sales_matrix.sum(axis=0)

    # Paramètres de style
    #plt.style.use('seaborn-whitegrid')
    plt.figure(figsize=(15, 6))

    # Tracés
    plt.plot(total_demand, label='Demand', color='dodgerblue', linewidth=1.5)
    plt.plot(total_forecast, label='Forecast', color='orangered', linewidth=1.5)
    plt.plot(total_lost_sales, label='Lost Sales', color='limegreen', linewidth=1.2)
    plt.plot(total_inventory, label='Average Inventory', color='saddlebrown', linewidth=1.5)

    # Améliorations visuelles
    plt.title('Daily Aggregated Metrics', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Units', fontsize=12)
    plt.xticks(rotation=45)
    plt.tick_params(axis='x', labelsize=8)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right', frameon=True, framealpha=0.9)

    # Limiter les xticks pour lisibilité
    xtick_interval = 60  # afficher 1 tick tous les 60 jours
    plt.xticks(ticks=range(0, len(total_demand), xtick_interval),
            labels=total_demand.index[::xtick_interval], rotation=45)

    plt.tight_layout()
    plt.show()
    
    
def plot_smooth_n_days(demand_matrix, forecast_matrix, lost_sales, inventory_begin, inventory_end, rolling_window):
    # Transform in pd.DataFrame
    lost_sales_matrix = pd.DataFrame(lost_sales, index=demand_matrix.index, columns=demand_matrix.columns)
    inventory_begin_matrix = pd.DataFrame(inventory_begin, index=demand_matrix.index, columns=demand_matrix.columns)
    inventory_end_matrix = pd.DataFrame(inventory_end, index=demand_matrix.index, columns=demand_matrix.columns)

    # Scrap the first 7 days for the plot  
    demand_matrix = demand_matrix.iloc[:, 7:]
    forecast_matrix = forecast_matrix.iloc[:, 7:]
    lost_sales_matrix = lost_sales_matrix.iloc[:, 7:]
    inventory_begin_matrix = inventory_begin_matrix.iloc[:, 7:]
    inventory_end_matrix = inventory_end_matrix.iloc[:, 7:]

    # Agrégation
    total_demand = demand_matrix.sum(axis=0)
    total_forecast = forecast_matrix.sum(axis=0)
    total_inventory = (inventory_begin_matrix.sum(axis=0) + inventory_end_matrix.sum(axis=0)) / 2
    total_lost_sales = lost_sales_matrix.sum(axis=0)

    # Moyennes glissantes (n jours)
    smoothed_demand = total_demand.rolling(window=rolling_window).mean()
    smoothed_forecast = total_forecast.rolling(window=rolling_window).mean()
    smoothed_inventory = total_inventory.rolling(window=rolling_window).mean()
    smoothed_lost_sales = total_lost_sales.rolling(window=rolling_window).mean()
    
    
    # Paramètres de style
    #plt.style.use('seaborn-whitegrid')
    plt.figure(figsize=(15, 6))

    # Tracés
    plt.plot(smoothed_demand, label='Demand (Smoothed)', color='dodgerblue')
    plt.plot(smoothed_forecast, label='Forecast (Smoothed)', color='orangered')
    plt.plot(smoothed_lost_sales, label='Lost Sales (Smoothed)', color='limegreen')
    plt.plot(smoothed_inventory, label='Avg Inventory (Smoothed)', color='saddlebrown')

    # Améliorations visuelles
    plt.title('Smoothed Metrics ('+str(rolling_window)+'-Day Rolling)', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Units', fontsize=12)
    plt.xticks(rotation=45)
    plt.tick_params(axis='x', labelsize=8)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right', frameon=True, framealpha=0.9)
    
    # Limiter les xticks pour lisibilité
    xtick_interval = 60  # afficher 1 tick tous les 60 jours
    plt.xticks(ticks=range(0, len(total_demand), xtick_interval),
            labels=total_demand.index[::xtick_interval], rotation=45)
    plt.tight_layout()
    plt.show()

def plot_histogram(df):
    means, stds = analyse_data(df)

    # Écart type global sur les valeurs aplaties
    sd = np.std(df.values.flatten())

    # Filtrage des valeurs <= 2*écart-type
    filtered_data = df[df <= 2 * sd]

    # Histogramme
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(filtered_data.values.flatten(), bins=50)
    mean = means.mean()
    plt.axvline(x=mean, color='red', linestyle='--')
    plt.text(mean, 0, 'mu', color='red', verticalalignment='bottom')
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title('Histogram of Values in the Demand')

    # Q-Q plot des log(données)
    plt.subplot(1, 2, 2)
    flattened = filtered_data.values.flatten()
    log_data = np.log(flattened[flattened > 0])  # filtrer les zéros ou négatifs
    stats.probplot(log_data, dist="norm", plot=plt)
    plt.title("Q-Q Plot of log(Values)")

    plt.tight_layout()
    plt.show()

def plot_pareto(policy_results):
    # Sort the results by average inventory
    sorted_items = sorted(policy_results.items(), key=lambda x: x[1][1])
    Variables = [item[0] for item in sorted_items]
    fill_rates = [item[1][0] for item in sorted_items]
    avg_inventories = [item[1][1] for item in sorted_items]
    
    plt.figure(figsize=(8, 6))
    plt.plot(avg_inventories, fill_rates, marker='o', linestyle='-', color='blue', label='Frontier')
    
    # Annotate the points with variables values
    for inv, fr, name in zip(avg_inventories, fill_rates, Variables):
        plt.annotate(name, (inv, fr), textcoords="offset points", xytext=(0,5), ha='center')
    
    plt.xlabel("Fill Rate")
    plt.ylabel("Average Inventory")
    plt.title("Strategy Results (Average Inventory vs. Fill Rate)")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_results(results):
    num_results = len(results)
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i % 10) for i in range(num_results)]
    
    plt.figure(figsize=(15, 10))

    for idx, (key, value) in enumerate(results.items()):
        # Sort the results by average inventory
        sorted_items = sorted(value.items(), key=lambda x: x[1][1])
        Variables = [item[0] for item in sorted_items]
        fill_rates = [item[1][0] for item in sorted_items]
        avg_inventories = [item[1][1] for item in sorted_items]
        
        plt.plot(avg_inventories, fill_rates, marker='o', linestyle='-', color=colors[idx], label=key)
        plt.annotate(key, 
                     xy=(avg_inventories[-1], fill_rates[-1]), 
                     xytext=(10, 0), 
                     textcoords='offset points', 
                     fontsize=9, 
                     color=colors[idx])
        
        # Annotate the points with variables values
        """ for inv, fr, name in zip(avg_inventories, fill_rates, Variables):
            plt.annotate(name, (inv, fr), textcoords="offset points", xytext=(0,5), ha='center') """

    plt.xlabel("Average Inventory")
    plt.ylabel("Fill Rate")
    plt.title("Results: Average Inventory vs Fill Rate")
    plt.legend(title="Policies")
    plt.grid(True)
    plt.show()

def output(function_name: str, fill_rate: float, avg_inventory: float, ratio: float, output_csv: str):
    """
    Update or create a line in the output CSV file for the given function name with the provided metrics.
    
    Parameters:
        function_name (str): Name of the function.
        fill_rate (float): Fill rate value.
        avg_inventory (float): Average inventory value.
        ratio (float): Ratio of days with perfect service.
        output_csv (str): Path to the output CSV file.
    """
    # Load the existing CSV file or create a new DataFrame if it doesn't exist
    try:
        df_output = pd.read_csv(output_csv, index_col='Function Name')
    except FileNotFoundError:
        df_output = pd.DataFrame(columns=['Function Name', 'Fill Rate', 'Avg Inventory', 'Ratio'])
        df_output.set_index('Function Name', inplace=True)
    
    # Update or create the row for the given function name
    df_output.loc[function_name] = [fill_rate, avg_inventory, ratio]
    
    # Save the updated DataFrame back to the CSV file
    df_output.to_csv(output_csv)

# Example usage
output('FC_avg_2_days', 0.95, 1500, 0.85, 'output_metrics.csv')
def pipeline(forecast_func=None, fc_factor=None,
             ss_func=SS_cst, ss_factor=0,
             up_to_level_func=None, X=None, min_fill_rate=None,
             compute_stock_func=None, order_rate=1, lead_time=2,
             plot= False):
    df = load_data(data_path)
    df = scraping_high_zeroes_products(df, n=2000)
    demand_matrix = preprocess_data(df)
    #demand_matrix = demand_matrix.iloc[:1, :]# A RETIRER
    _, n_days = demand_matrix.shape
    forecast_matrix = forecast_func(demand_matrix, fc_factor)
    safety_stock_matrix = ss_func(demand_matrix, forecast_matrix, ss_factor, order_rate+lead_time)
    up_to_level_matrix = up_to_level_func(demand_matrix, forecast_matrix, safety_stock_matrix, X, min_fill_rate, order_rate+lead_time)
    inventory_begin, inventory_end, _, _, _, lost_sales = compute_stock_func(up_to_level_matrix.to_numpy(), demand_matrix.to_numpy(), order_rate)
    fill_rate, avg_inventory, perfect_service_days_ratio = compute_results(demand_matrix, lost_sales, inventory_begin, inventory_end, n_days)

    if plot:
        plot_stocks(demand_matrix, forecast_matrix, lost_sales, inventory_begin, inventory_end)
        plot_smooth_n_days(demand_matrix, forecast_matrix, lost_sales, inventory_begin, inventory_end, 7)
        plot_smooth_n_days(demand_matrix, forecast_matrix, lost_sales, inventory_begin, inventory_end, 30)

    print(f"Fill rate: {fill_rate * 100:.2f}%")
    print(f"Average inventory: {avg_inventory}")
    print(f"Average proportion of products not in stock: {perfect_service_days_ratio * 100:.2f}%")

    return fill_rate, avg_inventory, perfect_service_days_ratio

results = {}
ss_factors = [0, 0.2, 0.4, 0.5, 0.75, 1, 1.25, 1.5, 2, 3]
CSLs = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99]
n_rolling = [3, 7, 14]
n_same = [4, 8]

results['theoretical'] = compute_pareto(
        _forecast_func=FC_avg_n_days, _fc_factor=3,
        _up_to_level_func=up_to_level_normal_demand, _min_fill_rate=CSLs,
        _compute_stock_func=compute_stock, _order_rate=7
    )
for n in n_rolling:
    results['moving_avg_'+ str(n) +'_dynamic'] = compute_pareto(
        _forecast_func=FC_avg_n_days, _fc_factor=n,
        _ss_func = SS_on_forecast, _ss_factor=ss_factors,
        _up_to_level_func=up_to_level_on_forecast,
        _compute_stock_func=compute_stock, _order_rate=7
    )
for n in n_same:
    results['same_days_avg_'+ str(n) +'_dynamic'] = compute_pareto(
        _forecast_func=FC_avg_n_same_days, _fc_factor=n,
        _ss_func = SS_on_forecast, _ss_factor=ss_factors,
        _up_to_level_func=up_to_level_on_forecast,
        _compute_stock_func=compute_stock, _order_rate=7
    )
    results['same_days_med_'+ str(n) +'_dynamic'] = compute_pareto(
        _forecast_func=FC_med_n_same_days, _fc_factor=n,
        _ss_func = SS_on_forecast, _ss_factor=ss_factors,
        _up_to_level_func=up_to_level_on_forecast,
        _compute_stock_func=compute_stock, _order_rate=7
    )

