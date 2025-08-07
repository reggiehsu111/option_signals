import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from typing import Optional, Dict, Any
from dotenv import load_dotenv
import os
from zoneinfo import ZoneInfo
import logging

logger = logging.getLogger(__name__)


class PolygonOHLCDataloader:
    """
    Data loader for OHLC (Open, High, Low, Close) price data from Polygon.io API.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.polygon.io",
        time_granularity: str = "1d",
    ):
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

    def generate(
        self,
        start_time: str,
        end_time: str,
        ticker: str = "I:SPX",
        src: str = "c",
        sort: str = "asc",
        limit: int = 5000,
    ) -> pd.DataFrame:
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
            "1d": ("1", "day"),
        }

        if self.time_granularity not in granularity_map:
            raise ValueError(f"Unsupported time_granularity: {self.time_granularity}")

        timespan, multiplier = granularity_map[self.time_granularity]
        ohlc_data = self._fetch_real_ohlc(
            ticker, start_time, end_time, timespan, multiplier, sort, limit
        )
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
        Fetch real OHLC data from Polygon.io API, handling pagination to retrieve all data up to end_time.

        Args:
            ticker (str): The ticker symbol
            start_time (str): Start time in 'YYYY-MM-DD' or millisecond timestamp
            end_time (str): End time in 'YYYY-MM-DD' or millisecond timestamp
            timespan (str): Time range multiplier (e.g., '1', '5')
            multiplier (str): Time unit (e.g., 'minute', 'hour', 'day')
            sort (str): Sort order ('asc' or 'desc')
            limit (int): Maximum number of results per API call

        Returns:
            pd.DataFrame: DataFrame with OHLC data and timestamp index
        """
        # Convert start_time and end_time to 'YYYY-MM-DD' if they are millisecond timestamps
        try:
            if start_time.isdigit():
                start_time = datetime.fromtimestamp(int(start_time) / 1000).strftime('%Y-%m-%d')
            if end_time.isdigit():
                end_time = datetime.fromtimestamp(int(end_time) / 1000).strftime('%Y-%m-%d')
        except ValueError as e:
            raise ValueError(f"Invalid time format for start_time or end_time: {e}")

        # Validate that start_time and end_time are in 'YYYY-MM-DD' format
        try:
            pd.to_datetime(start_time)
            pd.to_datetime(end_time)
        except ValueError as e:
            raise ValueError(f"start_time and end_time must be in 'YYYY-MM-DD' format: {e}")

        # Initialize variables for pagination
        all_results = []
        current_start_time = start_time
        end_time_dt = pd.to_datetime(end_time).tz_localize('US/Eastern')
        last_fetched_timestamp = None
        
        while True:
            # Construct the API URL
            url = f"{self.base_url}/v2/aggs/ticker/{ticker}/range/{timespan}/{multiplier}/{current_start_time}/{end_time}"
            params = {
                'apiKey': self.api_key,
                'sort': sort,
                'limit': limit
            }
            logger.info(f"Fetching data from {url} with params {params}")
            print(f"Fetching data between {current_start_time} and {end_time}")
            
            try:
                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                if data['status'] != 'OK':
                    raise ValueError(f"API returned status: {data['status']} - {data.get('error', 'No error message provided')}")
                
                # Extract results
                results = data.get('results', [])
                if not results:
                    logger.info(f"No more results for {current_start_time} to {end_time}")
                    break
                
                # Convert to DataFrame
                df = pd.DataFrame(results)
                # Convert millisecond timestamp to datetime and set to EST timezone
                df['timestamp'] = pd.to_datetime(df['t'], unit='ms').dt.tz_localize('UTC').dt.tz_convert(ZoneInfo('US/Eastern'))
                logger.info(f"Fetched {len(df)} rows from {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
                
                # Append to all results
                all_results.append(df)
                
                # Get the last timestamp
                last_timestamp = df['timestamp'].iloc[-1]
                
                # Check if we’re stuck fetching the same data
                if last_fetched_timestamp == last_timestamp:
                    logger.warning(f"Same timestamp {last_timestamp} fetched again, stopping to prevent infinite loop")
                    break
                last_fetched_timestamp = last_timestamp
      
                
                # Update start_time to the next day after the last timestamp
                current_start_time = (last_timestamp + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                
                # Ensure we don’t overshoot end_time
                if pd.to_datetime(current_start_time).tz_localize('US/Eastern') >= end_time_dt:
                    logger.info(f"Next start time {current_start_time} exceeds end time {end_time}, stopping")
                    break
                    
            except requests.exceptions.HTTPError as e:
                raise Exception(f"Failed to fetch data from Polygon.io: {e} - URL: {url}")
            except requests.exceptions.RequestException as e:
                raise Exception(f"Failed to fetch data from Polygon.io: {e} - URL: {url}")
        
        # Combine all results into a single DataFrame
        if not all_results:
            logger.info("No data fetched, returning empty DataFrame")
            return pd.DataFrame(columns=['o', 'h', 'l', 'c'])
        
        final_df = pd.concat(all_results, ignore_index=True)
        # Ensure no duplicate timestamps
        final_df = final_df.drop_duplicates(subset=['t'])
        # Convert timestamp again to ensure consistency
        final_df['timestamp'] = pd.to_datetime(final_df['t'], unit='ms').dt.tz_localize('UTC').dt.tz_convert(ZoneInfo('US/Eastern'))
        final_df = final_df.sort_values('timestamp') if sort == 'asc' else final_df.sort_values('timestamp', ascending=False)
        final_df = final_df.set_index('timestamp')
        
        # Filter to ensure data does not exceed end_time
        final_df = final_df[final_df.index <= end_time_dt]
        
        logger.info(f"Final DataFrame: {len(final_df)} rows from {final_df.index.min()} to {final_df.index.max()}")
        return final_df[['o', 'h', 'l', 'c']]

if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv("POLYGON_API_KEY")
    # Example usage with real API
    dataloader = PolygonOHLCDataloader(api_key=api_key, time_granularity="1h")
    ohlc_data = dataloader.generate(
        start_time="2018-01-01",
        end_time="2024-01-10",
    )
    print("Real OHLC Data:")
    print(ohlc_data)
