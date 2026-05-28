import pandas as pd
from datetime import datetime, timedelta
import pickle
from tqdm import tqdm

""" 
    Creates calander_dates file in GTFS Data folder by using calander.txt file
"""

with open(f'./parameters_entered.txt', 'rb') as file:
    parameter_files = pickle.load(file)
BUILD_TRANSFER, NETWORK_NAME, BUILD_TBTR_FILES, BUILD_TRANSFER_PATTERNS_FILES, BUILD_CSA = parameter_files

# Load calendar.txt
calendar = pd.read_csv(f"./Data/GTFS/{NETWORK_NAME}/gtfs_o/calendar.txt")


# Convert date columns
calendar['start_date'] = pd.to_datetime(calendar['start_date'], format='%Y%m%d')
calendar['end_date'] = pd.to_datetime(calendar['end_date'], format='%Y%m%d')

# Weekday mapping
weekday_cols = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
weekday_map = {0: 'monday', 1: 'tuesday', 2: 'wednesday', 3: 'thursday', 4: 'friday', 5: 'saturday', 6: 'sunday'}

# Prepare list to collect rows
calendar_dates_rows = []

# Expand each service_id's range into dates
for _, row in tqdm(calendar.iterrows()):
    current_date = row['start_date']
    while current_date <= row['end_date']:
        weekday = weekday_map[current_date.weekday()]
        if row[weekday] == 1:
            calendar_dates_rows.append({
                'service_id': row['service_id'],
                'date': current_date.strftime('%Y%m%d'),
                'exception_type': 1  # 1 = service added
            })
        current_date += timedelta(days=1)

# Create DataFrame
calendar_dates = pd.DataFrame(calendar_dates_rows)

# Optional: Save it as a GTFS-style text file
calendar_dates.to_csv(f'./Data/GTFS/{NETWORK_NAME}/gtfs_o/calendar_dates.txt', index=False)
