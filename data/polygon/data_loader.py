import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from typing import Optional, Dict, Any
from dotenv import load_dotenv
import os
from zoneinfo import ZoneInfo

class PolygonOHLCDataloader:
    """
    Data loader for OHLC (Open, High, Low, Close) price data from Polygon.io API.
    """
    
    def __init__(self, 
                 api_key: str,
                 base_url: str = "https://api.polygon.io",
                 time_granularity: str = "1d"):
        """
        Initialize the OHLC data loader.

        Args:
            api_key (str): Polygon.io API key (required)
            base_url (str): Base URL for Polygon.io API
        """
        if not api_key:
            raise ValueError("API key is required for PolygonOHLCDataloader")
        
        if time_granularity not in ["1m", "5m", "1h", "4h", "1d"]:
            raise ValueError(
                "time_granularity must be one of: '1m', '5m', '1h', '4h', '1d'"
            )
        
        self.api_key = api_key
        self.base_url = base_url
        self.time_granularity = time_granularity

    def generate(self, 
                start_time: str,
                end_time: str,
                ticker: str = "I:SPX",
                src: str = "close",
                sort: str = "asc",
                limit: int = 5000) -> pd.DataFrame:
        """
        Get OHLC data for a given ticker and time range using the configured time_granularity.

        Args:
            start_time (str): Start time in format 'YYYY-MM-DD' or millisecond timestamp
            end_time (str): End time in format 'YYYY-MM-DD' or millisecond timestamp
            ticker (str): The ticker symbol (e.g., 'SPY', 'AAPL', 'I:SPX', 'I:VIX') (default: "I:SPX")
            src (str): Data source column to return (eg. "close", "open", "high", "low") (default: "close") 
            sort (str): Sort order - 'asc' or 'desc' (default: "asc")
            limit (int): Maximum number of results (default: 5000)

        Returns:
            pd.DataFrame: DataFrame with OHLC data and timestamp index
        """
        # Map time_granularity to timespan and multiplier
        granularity_map = {
            "1m": ("1", "minute"),
            "5m": ("5", "minute"),
            "1h": ("1", "hour"),
            "4h": ("4", "hour"),
            "1d": ("1", "day")
        }
        
        if self.time_granularity not in granularity_map:
            raise ValueError(f"Unsupported time_granularity: {self.time_granularity}")
        
        timespan, multiplier = granularity_map[self.time_granularity]
        ohlc_data = self._fetch_real_ohlc(ticker, start_time, end_time, timespan, multiplier, sort, limit)
        return ohlc_data[src]

    def _fetch_real_ohlc(self, 
                        ticker: str,
                        start_time: str,
                        end_time: str,
                        timespan: str,
                        multiplier: str,
                        sort: str,
                        limit: int) -> pd.DataFrame:
        """
        Fetch real OHLC data from Polygon.io API.
        """
        # Construct the API URL based on the documentation
        url = f"{self.base_url}/v2/aggs/ticker/{ticker}/range/{timespan}/{multiplier}/{start_time}/{end_time}"
        
        params = {
            'apiKey': self.api_key,
            'sort': sort,
            'limit': limit
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data['status'] != 'OK':
                raise ValueError(f"API returned status: {data['status']}")
            
            # Extract results
            results = data.get('results', [])
            if not results:
                return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
            
            # Convert to DataFrame
            df = pd.DataFrame(results)
            # Convert millisecond timestamp to datetime and set to EST timezone
            df['timestamp'] = pd.to_datetime(df['t'], unit='ms').dt.tz_localize('UTC').dt.tz_convert(ZoneInfo('US/Eastern'))
            df = df.set_index('timestamp')
            
            # Rename columns to match standard OHLC naming
            df = df.rename(columns={
                'o': 'open',
                'h': 'high', 
                'l': 'low',
                'c': 'close',
                'v': 'volume'
            })
            
            # Select only OHLC columns
            ohlc_columns = ['open', 'high', 'low', 'close']
            return df[ohlc_columns]
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to fetch data from Polygon.io: {e}")


if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv("POLYGON_API_KEY")
    # Example usage with real API
    dataloader = PolygonOHLCDataloader(api_key=api_key, time_granularity="5m")
    ohlc_data = dataloader.generate(
        start_time="2024-01-01",
        end_time="2024-01-10",
    )
    print("Real OHLC Data:")
    print(ohlc_data)
    