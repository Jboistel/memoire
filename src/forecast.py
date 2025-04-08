import utils.py

def FC_avg_3_days(demand_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the forecast by averaging the last 3 days for each product.
    
    Parameters:
        demand_matrix (pd.DataFrame): Demand matrix.
    
    Returns:
        pd.DataFrame: Forecast matrix (rounded up).
    """
    forecast_matrix = demand_matrix.T.rolling(window=3).mean().shift(1).T
    return forecast_matrix

def FC_avg_7_days(demand_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the forecast by averaging the last 7 days for each product.
    
    Parameters:
        demand_matrix (pd.DataFrame): Demand matrix.
    
    Returns:
        pd.DataFrame: Forecast matrix (rounded up).
    """
    forecast_matrix = demand_matrix.T.rolling(window=7).mean().shift(1).T
    return forecast_matrix

def FC_avg_30_days(demand_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the forecast by averaging the last 30 days for each product.
    
    Parameters:
        demand_matrix (pd.DataFrame): Demand matrix.
    
    Returns:
        pd.DataFrame: Forecast matrix (rounded up).
    """
    forecast_matrix = demand_matrix.T.rolling(window=30).mean().shift(1).T
    return forecast_matrix

def FC_avg_4_same_days(demand_matrix: pd.DataFrame) -> pd.DataFrame:
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
        # Need at least 5 weeks to use the previous 4 weeks
        if len(idx) < 5:
            continue
        # Create a sliding window view for the previous 4 weeks
        slices = np.stack([demand_np[:, idx[i: i + len(idx) - 4]] for i in range(4)], axis=-1)
        rolling_mean = np.mean(slices, axis=-1)
        target_indices = idx[4:]  # Days to forecast
        weekly_forecast_np[:, target_indices] = rolling_mean
        
    weekly_forecast_matrix = pd.DataFrame(
        weekly_forecast_np,
        index=demand_matrix.index,
        columns=demand_matrix.columns
    )
    return weekly_forecast_matrix