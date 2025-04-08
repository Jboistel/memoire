import pandas as pd

def load_data(csv_path: str, pkl_path: str) -> pd.DataFrame:
    """
    Load the CSV file, save it as a pickle file, and then reload it.
    
    Parameters:
        csv_path (str): Path to the CSV file.
        pkl_path (str): Path where the pickle file will be saved.
    
    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    df = pd.read_csv(csv_path)
    df.to_pickle(pkl_path)
    df = pd.read_pickle(pkl_path)
    return df

def scraping_high_zeroes_products(df: pd.DataFrame, nb_product: int) -> pd.DataFrame:
    # Count the number of zeroes in each row
    df['zero_count'] = (df == 0).sum(axis=1)
    
    # Sort the DataFrame by the number of zeroes in ascending order
    df_sorted = df.sort_values(by='zero_count')
    
    # Keep only the top n rows with the least number of zeroes
    df_top = df_sorted.head(nb_product)
    
    # Drop the 'zero_count' column as it is no longer needed
    df_top = df_top.drop(columns=['zero_count'])
    
    return df_top

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
    calendar = pd.read_csv('/home/onyx/Documents/Mémoire/memoire/data/calendar.csv')

    # Get the date columns from the calendar dataframe
    date_columns = calendar['date']
    demand_matrix.columns = date_columns[0:demand_matrix.shape[1]]
    
    
    return demand_matrix

def analyse_data(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    means = df.mean(axis=1)
    stds = df.std(axis=1)
    
    return means, stds
