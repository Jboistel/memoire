import pandas as pd
import numpy as np# Read the CSV file and create a DataFrame
df = pd.read_csv('data/sales_train_evaluation.csv')

df.to_pickle('data/sales_train_evaluation.pkl')
df = pd.read_pickle('data/sales_train_evaluation.pkl')
df
# Supprimer les colonnes non nécessaires pour la matrice
df_demand = df.drop(columns=['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'])

# Transformer en format pivoté : produits (id) en lignes, jours (d_1, d_2, ...) en colonnes
demand_matrix = df_demand.set_index('id')
print(demand_matrix.dtypes)
print(demand_matrix.info(memory_usage='deep'))
# Get the column names except for "id"
columns = demand_matrix.columns[demand_matrix.columns != 'id']

# Iterate over the columns and convert the data type
for column in columns:
    demand_matrix[column] = pd.to_numeric(demand_matrix[column], downcast='integer')

# Check the new data types
print(demand_matrix.dtypes)
print(demand_matrix.info(memory_usage='deep'))
# Afficher un aperçu
print("Matrice des demandes :")
print(demand_matrix.head())
def FC_avg_3_days():
    """Computes the average of the last 3 days for each product"""
    forecast_matrix = demand_matrix.T.rolling(window=3).mean().shift(1).T

    # Rounds the forecast up to the next integer
    forecast_matrix = np.ceil(forecast_matrix)
    return forecast_matrix

def FC_avg_7_days():
    """Computes the average of the last 7 days for each product"""
    forecast_matrix = demand_matrix.T.rolling(window=7).mean().shift(1).T # TODO: Check if the shift is correct

    # Rounds the forecast up to the next integer
    forecast_matrix = np.ceil(forecast_matrix)
    return forecast_matrix

def FC_avg_4_same_days():
    """Computes the average of the same 4 days in the previous 4 weeks for each product"""
    # Convertir en matrice NumPy
    demand_np = demand_matrix.to_numpy()
    n_products, n_days = demand_np.shape

    # Initialiser le tableau forecast
    weekly_forecast_np = np.full((n_products, n_days), np.nan)

    # Pour chaque jour de la semaine (offset = 0 à 6)
    for offset in range(7):
        # Sélectionner les colonnes alignées sur le même jour de semaine
        idx = np.arange(offset, n_days, 7)

        # On a besoin d'au moins 5 semaines pour pouvoir prendre les 4 précédentes
        if len(idx) < 5:
            continue

        # Créer un tableau 3D glissant : shape = (n_products, len(idx)-4, 4)
        slices = np.stack([
            demand_np[:, idx[i: i + len(idx) - 4]]
            for i in [0, 1, 2, 3]
        ], axis=-1)

        # Moyenne sur les 4 valeurs précédentes
        rolling_mean = np.mean(slices, axis=-1)

        # Replacer les résultats dans les bonnes positions
        target_indices = idx[4:]  # Correspond aux jours à prédire
        weekly_forecast_np[:, target_indices] = rolling_mean

    weekly_forecast_matrix = pd.DataFrame(
    np.ceil(weekly_forecast_np),  # arrondi vers le haut
    index=demand_matrix.index,
    columns=demand_matrix.columns
    )
    return weekly_forecast_matrix

forecast_matrix = FC_avg_4_same_days()

# Afficher un aperçu
print("Matrice des prévisions (Forecast) :")
print(forecast_matrix.head())

up_to_level_matrix = forecast_matrix.T.shift(-2).rolling(window=3).sum().T
# Afficher un aperçu
print("Matrice du niveau à atteindre (up_to_level_matrix) :")
print(up_to_level_matrix.head())
def compute_stock(up_to_level_matrix):
    # Conversion vers arrays
    demand = demand_matrix.to_numpy()
    up_to_level = up_to_level_matrix.to_numpy()
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

    # Boucle vectorisée jour par
    # Conversion vers arrays
    demand = demand_matrix.to_numpy()
    up_to_level = up_to_level_matrix.to_numpy()
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
        T2[:, day] = np.maximum(up_to_level[:, day] - inventory_end[:, day] - T1[:, day], 0)
        sales[:, day] = inventory_begin[:, day] - inventory_end[:, day]
        lost_sales[:, day] = np.maximum(demand[:, day] - sales[:, day], 0)

    # Reconvertir en DataFrames si besoin
    columns = demand_matrix.columns
    index = demand_matrix.index

    inventory_begin_matrix = pd.DataFrame(inventory_begin, index=index, columns=columns)
    inventory_end_matrix = pd.DataFrame(inventory_end, index=index, columns=columns)
    T1_matrix = pd.DataFrame(T1, index=index, columns=columns)
    T2_matrix = pd.DataFrame(T2, index=index, columns=columns)
    sales_matrix = pd.DataFrame(sales, index=index, columns=columns)
    lost_sales_matrix = pd.DataFrame(lost_sales, index=index, columns=columns)

    return inventory_begin_matrix, inventory_end_matrix, T1_matrix, T2_matrix, sales_matrix, lost_sales_matrix
    # Afficher un aperçu
print("Inventory begin:")
print(inventory_begin_matrix.head())

print("Inventory end:")
print(inventory_end_matrix.head())

print("T1:")
print(T1_matrix.head())

print("T2:")
print(T2_matrix.head())

print("Sales:")
print(sales_matrix.head())

print("Lost sales:")
print(lost_sales_matrix.head())

def compute_results(demand_matrix, forecast_matrix, up_to_level_matrix):
    inventory_begin_matrix, inventory_end_matrix, T1_matrix, T2_matrix, sales_matrix, lost_sales_matrix = compute_stock(up_to_level_matrix)
    fill_rate = 1 - lost_sales.sum().sum() / demand.sum().sum()
    avg_inventory = (inventory_begin_matrix.sum().sum() + inventory_end_matrix.sum().sum()) / 2 // n_days
    perfect_service_days_ratio = (lost_sales_matrix.sum(axis=0) == 0).mean()