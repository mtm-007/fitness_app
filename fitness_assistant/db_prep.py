import os
import requests
import pandas as pd
from tqdm.auto import tqdm
from dotenv import load_dotenv

from db import init_db

load_dotenv()


if __name__ == "__main__":
    print("Initializing database...")
    init_db()
    print("Initializing database...successfully!")
   