import pandas as pd
import numpy as np
from datetime import datetime


class MockDataloader:
    def __init__(self, start_time: str, end_time: str, time_granularity: str):
        """
        Initialize the MockDataloader with time range and granularity.

        Args:
            start_time (str): Start time in format 'YYYY-MM-DD HH:MM:SS'
            end_time (str): End time in format 'YYYY-MM-DD HH:MM:SS'
            time_granularity (str): Frequency of data points, e.g., '1m', '5m', '1h', '4h', '1d'
        """
        self.start_time = pd.to_datetime(start_time)
        self.end_time = pd.to_datetime(end_time)
        self.time_granularity = time_granularity

        # Validate inputs
        if self.start_time >= self.end_time:
            raise ValueError("start_time must be before end_time")
        if self.time_granularity not in ["1m", "5m", "1h", "4h", "1d"]:
            raise ValueError(
                "time_granularity must be one of: '1m', '5m', '1h', '4h', '1d'"
            )

    def generate(self) -> pd.Series:
        """
        Generate a pandas Series with mock data spanning from start_time to end_time
        with specified time granularity.

        Returns:
            pd.Series: Series with datetime index and random float values
        """
        # Create date range with specified granularity
        date_rng = pd.date_range(
            start=self.start_time, end=self.end_time, freq=self.time_granularity
        )

        # Generate random data
        data = np.random.normal(loc=100, scale=10, size=len(date_rng))

        # Create Series
        series = pd.Series(data, index=date_rng, name="mock_data")

        return series


if __name__ == "__main__":
    dataloader = MockDataloader(
        start_time="2025-01-01 00:00:00",
        end_time="2025-02-01 16:00:00",
        time_granularity="1h",
    )
    print(dataloader.generate())