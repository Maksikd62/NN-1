import pandas as pd
import numpy as np

def load_monthly_data(file_path='Energy_Consumption_by_Month.csv'):
    df = pd.read_csv(file_path)
    df['month'] = df['month'].str.strip()
    months_map = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4,
        'May': 5, 'June': 6, 'July': 7, 'August': 8,
        'September': 9, 'October': 10, 'November': 11, 'December': 12
    }
    df['month_num'] = df['month'].map(months_map)
    df = df.dropna()
    X = df['month_num'].values.reshape(-1, 1)
    y = df['consumption_kwh'].values
    return X, y
