import os

# 1. Reproducibility Seed
SEED = 42

# 2. File Paths (Relative)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW = os.path.join(BASE_DIR, 'data', 'raw')
DATA_PROCESSED = os.path.join(BASE_DIR, 'data', 'processed')

# 3. Raw File Names
CALENDAR_FILE = os.path.join(DATA_RAW, 'calendar.csv.gz')
REVIEWS_FILE = os.path.join(DATA_RAW, 'reviews.csv.gz')
LISTINGS_FILE = os.path.join(DATA_RAW, 'listings.csv.gz') 

# 4. Target Definition
TARGET = 'Occupancy_Status'