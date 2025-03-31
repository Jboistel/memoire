import numpy as np
import pandas as pd

def compute_stock(up_to_level, demand):
    """
    Compute various stock management indicators:
      - inventory_begin: stock at the beginning of the day
      - inventory_end: stock at the end of the day
      - T1: stock transferred from the previous day
      - T2: additional stock needed to reach the forecasted level
      - sales: actual sales
      - lost_sales: unmet demand (lost sales)
    
    Parameters:
        demand_matrix (pd.DataFrame): Demand matrix.
        up_to_level_matrix (pd.DataFrame): Matrix of levels to reach.
        initial_inventory (int): Initial inventory at day 0.
    
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
        T2[:, day] = np.nan_to_num(np.maximum(up_to_level[:, day] - inventory_end[:, day] - T1[:, day], 0))
        sales[:, day] = inventory_begin[:, day] - inventory_end[:, day]
        lost_sales[:, day] = np.maximum(demand[:, day] - sales[:, day], 0)

    """ # Reconvertir en DataFrames si besoin
    columns = demand_matrix.columns
    index = demand_matrix.index

    inventory_begin_matrix = pd.DataFrame(inventory_begin, index=index, columns=columns)
    inventory_end_matrix = pd.DataFrame(inventory_end, index=index, columns=columns)
    T1_matrix = pd.DataFrame(T1, index=index, columns=columns)
    T2_matrix = pd.DataFrame(T2, index=index, columns=columns)
    sales_matrix = pd.DataFrame(sales, index=index, columns=columns)
    lost_sales_matrix = pd.DataFrame(lost_sales, index=index, columns=columns) """

    return inventory_begin, inventory_end, T1, T2, sales, lost_sales
