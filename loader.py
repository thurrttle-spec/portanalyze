"""
Data loader module for loading financial data from various file formats.

This module provides the DataLoader class which can load data from CSV, Excel,
and JSON files with automatic format detection, date parsing, and validation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime
import json
import re
from .models import StockData, MarketData, ValidationResult


class DataLoaderError(Exception):
    """Base exception for data loader errors."""
    pass


class FileFormatError(DataLoaderError):
    """Raised when file format is invalid or unsupported."""
    pass


class DataValidationError(DataLoaderError):
    """Raised when loaded data fails validation."""
    pass


class DataLoader:
    """Loader for financial data from various file formats.
    
    Supports:
        - CSV files (.csv)
        - Excel files (.xlsx, .xls)
        - JSON files (.json)
        
    Features:
        - Automatic date format detection
        - File format validation
        - Missing value handling
        - Data quality validation
    """
    
    # Common date formats to try during auto-detection
    DATE_FORMATS = [
        '%Y-%m-%d',           # 2023-01-15
        '%d/%m/%Y',           # 15/01/2023
        '%m/%d/%Y',           # 01/15/2023
        '%Y/%m/%d',           # 2023/01/15
        '%d-%m-%Y',           # 15-01-2023
        '%m-%d-%Y',           # 01-15-2023
        '%Y%m%d',             # 20230115
        '%d.%m.%Y',           # 15.01.2023
        '%Y.%m.%d',           # 2023.01.15
        '%B %d, %Y',          # January 15, 2023
        '%d %B %Y',           # 15 January 2023
    ]
    
    def __init__(self):
        """Initialize the DataLoader."""
        self._loaded_data: Optional[pd.DataFrame] = None
        self._file_path: Optional[Path] = None
    
    def load_file(
        self,
        file_path: Union[str, Path],
        sheet_name: Optional[str] = None,
        date_column: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Load data from file with automatic format detection.
        
        Args:
            file_path: Path to the data file
            sheet_name: Sheet name for Excel files (optional)
            date_column: Name of the date column for parsing (optional)
            **kwargs: Additional arguments passed to the file reader
            
        Returns:
            DataFrame with loaded data
            
        Raises:
            FileNotFoundError: If file doesn't exist
            FileFormatError: If file format is invalid or unsupported
            DataLoaderError: If data loading fails
        """
        file_path = Path(file_path)
        self._file_path = file_path
        
        # Validate file exists
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Validate file format
        self._validate_file_format(file_path)
        
        # Load based on file extension
        suffix = file_path.suffix.lower()
        
        try:
            if suffix == '.csv':
                df = self._load_csv(file_path, **kwargs)
            elif suffix in ['.xlsx', '.xls']:
                df = self._load_excel(file_path, sheet_name, **kwargs)
            elif suffix == '.json':
                df = self._load_json(file_path, **kwargs)
            else:
                raise FileFormatError(
                    f"Unsupported file format: {suffix}. "
                    f"Supported formats: .csv, .xlsx, .xls, .json"
                )
            
            # Auto-detect and parse date column if specified
            if date_column and date_column in df.columns:
                df[date_column] = self._parse_dates(df[date_column])
                df = df.set_index(date_column)
            elif date_column:
                # Try to find date column by pattern
                detected_col = self._detect_date_column(df)
                if detected_col:
                    df[detected_col] = self._parse_dates(df[detected_col])
                    df = df.set_index(detected_col)
            
            self._loaded_data = df
            return df
            
        except Exception as e:
            if isinstance(e, (FileFormatError, DataLoaderError)):
                raise
            raise DataLoaderError(f"Failed to load file {file_path}: {str(e)}") from e
    
    def _validate_file_format(self, file_path: Path) -> None:
        """Validate file format and basic file integrity.
        
        Args:
            file_path: Path to validate
            
        Raises:
            FileFormatError: If file format is invalid
        """
        # Check file size
        file_size = file_path.stat().st_size
        if file_size == 0:
            raise FileFormatError(f"File is empty: {file_path}")
        
        # Check file extension
        suffix = file_path.suffix.lower()
        if suffix not in ['.csv', '.xlsx', '.xls', '.json']:
            raise FileFormatError(
                f"Unsupported file extension: {suffix}. "
                f"Supported: .csv, .xlsx, .xls, .json"
            )
        
        # Basic format validation
        if suffix == '.csv':
            self._validate_csv_format(file_path)
        elif suffix == '.json':
            self._validate_json_format(file_path)
    
    def _validate_csv_format(self, file_path: Path) -> None:
        """Validate CSV file format.
        
        Args:
            file_path: Path to CSV file
            
        Raises:
            FileFormatError: If CSV format is invalid
        """
        try:
            # Try to read first few lines
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline()
                if not first_line.strip():
                    raise FileFormatError("CSV file has no content")
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    first_line = f.readline()
                    if not first_line.strip():
                        raise FileFormatError("CSV file has no content")
            except Exception as e:
                raise FileFormatError(f"Cannot read CSV file: {str(e)}") from e
    
    def _validate_json_format(self, file_path: Path) -> None:
        """Validate JSON file format.
        
        Args:
            file_path: Path to JSON file
            
        Raises:
            FileFormatError: If JSON format is invalid
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                json.load(f)
        except json.JSONDecodeError as e:
            raise FileFormatError(f"Invalid JSON format: {str(e)}") from e
        except Exception as e:
            raise FileFormatError(f"Cannot read JSON file: {str(e)}") from e
    
    def _load_csv(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Load data from CSV file.
        
        Args:
            file_path: Path to CSV file
            **kwargs: Additional arguments for pd.read_csv
            
        Returns:
            DataFrame with loaded data
        """
        # Try UTF-8 first, then fall back to latin-1
        try:
            df = pd.read_csv(file_path, encoding='utf-8', **kwargs)
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='latin-1', **kwargs)
        
        return df
    
    def _load_excel(
        self,
        file_path: Path,
        sheet_name: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Load data from Excel file.
        
        Args:
            file_path: Path to Excel file
            sheet_name: Name of sheet to load (default: first sheet)
            **kwargs: Additional arguments for pd.read_excel
            
        Returns:
            DataFrame with loaded data
        """
        if sheet_name is None:
            sheet_name = 0  # Load first sheet
        
        df = pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)
        return df
    
    def _load_json(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Load data from JSON file.
        
        Args:
            file_path: Path to JSON file
            **kwargs: Additional arguments for pd.read_json
            
        Returns:
            DataFrame with loaded data
        """
        # Try to load as records format first
        try:
            df = pd.read_json(file_path, orient='records', **kwargs)
        except ValueError:
            # Try other orientations
            for orient in ['index', 'columns', 'values', 'split', 'table']:
                try:
                    df = pd.read_json(file_path, orient=orient, **kwargs)
                    break
                except (ValueError, KeyError):
                    continue
            else:
                raise FileFormatError("Cannot determine JSON orientation")
        
        return df
    
    def _detect_date_column(self, df: pd.DataFrame) -> Optional[str]:
        """Detect date column in DataFrame.
        
        Args:
            df: DataFrame to search
            
        Returns:
            Name of detected date column, or None if not found
        """
        # Common date column names
        date_patterns = [
            r'^date$', r'^datetime$', r'^timestamp$',
            r'^time$', r'^day$', r'^dt$', r'.*date.*'
        ]
        
        for col in df.columns:
            col_lower = str(col).lower()
            for pattern in date_patterns:
                if re.match(pattern, col_lower):
                    # Verify it actually contains date-like data
                    if self._looks_like_dates(df[col]):
                        return col
        
        return None
    
    def _looks_like_dates(self, series: pd.Series) -> bool:
        """Check if a series contains date-like values.
        
        Args:
            series: Series to check
            
        Returns:
            True if series appears to contain dates
        """
        # Sample first non-null value
        sample = series.dropna().head(5)
        if len(sample) == 0:
            return False
        
        for value in sample:
            # Try parsing as date
            parsed = self._try_parse_date(str(value))
            if parsed is not None:
                return True
        
        return False
    
    def _parse_dates(self, series: pd.Series) -> pd.Series:
        """Parse dates with automatic format detection.
        
        Args:
            series: Series with date strings
            
        Returns:
            Series with parsed datetime values
        """
        # If already datetime, return as is
        if pd.api.types.is_datetime64_any_dtype(series):
            return series
        
        # Try pandas automatic parsing first
        try:
            return pd.to_datetime(series)
        except (ValueError, TypeError):
            pass
        
        # Try each date format
        for date_format in self.DATE_FORMATS:
            try:
                parsed = pd.to_datetime(series, format=date_format)
                return parsed
            except (ValueError, TypeError):
                continue
        
        # Last resort: try parsing with infer_datetime_format
        try:
            return pd.to_datetime(series, infer_datetime_format=True)
        except Exception as e:
            raise DataLoaderError(
                f"Cannot parse dates. Tried {len(self.DATE_FORMATS)} formats. "
                f"Error: {str(e)}"
            ) from e
    
    def _try_parse_date(self, date_str: str) -> Optional[datetime]:
        """Try to parse a single date string.
        
        Args:
            date_str: String to parse
            
        Returns:
            Parsed datetime or None if parsing failed
        """
        for date_format in self.DATE_FORMATS:
            try:
                return datetime.strptime(date_str, date_format)
            except (ValueError, TypeError):
                continue
        return None
    
    def load_stock_data(
        self,
        file_path: Union[str, Path],
        ticker_columns: Optional[List[str]] = None,
        date_column: str = 'Date',
        sheet_name: Optional[str] = None,
        **kwargs
    ) -> Dict[str, StockData]:
        """Load stock data from file and create StockData instances.
        
        Args:
            file_path: Path to data file
            ticker_columns: List of column names for stock tickers
                          (if None, auto-detect columns ending with .JK)
            date_column: Name of the date column
            sheet_name: Sheet name for Excel files
            **kwargs: Additional arguments for file reader
            
        Returns:
            Dictionary mapping ticker symbols to StockData instances
            
        Raises:
            DataLoaderError: If loading or parsing fails
        """
        # Load raw data
        df = self.load_file(file_path, sheet_name=sheet_name, **kwargs)
        
        # Set date column as index if not already
        if date_column in df.columns:
            df[date_column] = self._parse_dates(df[date_column])
            df = df.set_index(date_column)
        elif not isinstance(df.index, pd.DatetimeIndex):
            # Try to detect date column
            detected = self._detect_date_column(df)
            if detected:
                df[detected] = self._parse_dates(df[detected])
                df = df.set_index(detected)
            else:
                raise DataLoaderError("Cannot find or detect date column")
        
        # Sort by date
        df = df.sort_index()
        
        # Auto-detect ticker columns if not provided
        if ticker_columns is None:
            ticker_columns = [col for col in df.columns if str(col).endswith('.JK')]
        
        if not ticker_columns:
            raise DataLoaderError("No ticker columns found or specified")
        
        # Create StockData instances
        stock_data_dict = {}
        
        for ticker in ticker_columns:
            if ticker not in df.columns:
                raise DataLoaderError(f"Column {ticker} not found in data")
            
            # Extract prices and handle missing values
            prices = df[ticker].copy()
            
            # Remove rows with missing prices
            prices = prices.dropna()
            
            if len(prices) == 0:
                raise DataLoaderError(f"No valid price data for {ticker}")
            
            # Calculate log returns
            returns = np.log(prices / prices.shift(1))
            
            # Create StockData instance
            try:
                stock_data = StockData(
                    ticker=ticker,
                    prices=prices,
                    returns=returns,
                    metadata={
                        'source_file': str(file_path),
                        'num_original_rows': len(df),
                        'num_valid_prices': len(prices)
                    }
                )
                stock_data_dict[ticker] = stock_data
            except ValueError as e:
                raise DataValidationError(
                    f"Validation failed for {ticker}: {str(e)}"
                ) from e
        
        return stock_data_dict
    
    def load_market_data(
        self,
        file_path: Union[str, Path],
        index_column: str = 'Market_Close',
        date_column: str = 'Date',
        risk_free_column: str = 'BI_Rate',
        index_name: str = 'Market Index',
        sheet_name: Optional[str] = None,
        **kwargs
    ) -> MarketData:
        """Load market index data and create MarketData instance.
        
        Args:
            file_path: Path to data file
            index_column: Name of the market index price column
            date_column: Name of the date column
            risk_free_column: Name of the risk-free rate column
            index_name: Name for the market index
            sheet_name: Sheet name for Excel files
            **kwargs: Additional arguments for file reader
            
        Returns:
            MarketData instance
            
        Raises:
            DataLoaderError: If loading or parsing fails
        """
        # Load raw data
        df = self.load_file(file_path, sheet_name=sheet_name, **kwargs)
        
        # Set date column as index if not already
        if date_column in df.columns:
            df[date_column] = self._parse_dates(df[date_column])
            df = df.set_index(date_column)
        elif not isinstance(df.index, pd.DatetimeIndex):
            detected = self._detect_date_column(df)
            if detected:
                df[detected] = self._parse_dates(df[detected])
                df = df.set_index(detected)
            else:
                raise DataLoaderError("Cannot find or detect date column")
        
        # Sort by date
        df = df.sort_index()
        
        # Validate required columns exist
        if index_column not in df.columns:
            raise DataLoaderError(f"Market index column '{index_column}' not found")
        
        if risk_free_column not in df.columns:
            raise DataLoaderError(f"Risk-free rate column '{risk_free_column}' not found")
        
        # Extract market prices
        prices = df[index_column].copy()
        prices = prices.dropna()
        
        if len(prices) == 0:
            raise DataLoaderError("No valid market index data")
        
        # Calculate log returns
        returns = np.log(prices / prices.shift(1))
        
        # Extract risk-free rate (use mean of the series)
        risk_free_series = df[risk_free_column].dropna()
        if len(risk_free_series) == 0:
            raise DataLoaderError("No valid risk-free rate data")
        
        # Convert percentage to decimal if needed (assume values > 1 are percentages)
        risk_free_mean = risk_free_series.mean()
        if risk_free_mean > 1:
            risk_free_rate = risk_free_mean / 100.0
        else:
            risk_free_rate = risk_free_mean
        
        # Create MarketData instance
        try:
            market_data = MarketData(
                index_name=index_name,
                prices=prices,
                returns=returns,
                risk_free_rate=risk_free_rate
            )
            return market_data
        except ValueError as e:
            raise DataValidationError(
                f"Validation failed for market data: {str(e)}"
            ) from e
    
    def load_financial_dataset(
        self,
        file_path: Union[str, Path],
        sheet_name: str = 'Data Gabungan Harian',
        date_column: str = 'Date',
        market_column: str = 'Market_Close',
        risk_free_column: str = 'BI_Rate',
        ticker_pattern: str = r'\.JK$',
        **kwargs
    ) -> Tuple[Dict[str, StockData], MarketData]:
        """Load complete financial dataset with stocks and market data.
        
        This is a convenience method that loads both stock and market data
        in a single call, suitable for the project's Excel file format.
        
        Args:
            file_path: Path to data file
            sheet_name: Sheet name for Excel files
            date_column: Name of the date column
            market_column: Name of the market index column
            risk_free_column: Name of the risk-free rate column
            ticker_pattern: Regex pattern to identify stock ticker columns
            **kwargs: Additional arguments for file reader
            
        Returns:
            Tuple of (stock_data_dict, market_data)
            
        Raises:
            DataLoaderError: If loading or parsing fails
        """
        # Load raw data
        df = self.load_file(file_path, sheet_name=sheet_name, **kwargs)
        
        # Set date column as index
        if date_column in df.columns:
            df[date_column] = self._parse_dates(df[date_column])
            df = df.set_index(date_column)
        
        # Sort by date
        df = df.sort_index()
        
        # Identify stock ticker columns
        ticker_columns = [
            col for col in df.columns
            if re.search(ticker_pattern, str(col))
        ]
        
        if not ticker_columns:
            raise DataLoaderError(
                f"No stock ticker columns found matching pattern: {ticker_pattern}"
            )
        
        # Load market data
        market_data = self._extract_market_data_from_df(
            df, market_column, risk_free_column
        )
        
        # Load stock data
        stock_data_dict = self._extract_stock_data_from_df(
            df, ticker_columns, file_path
        )
        
        return stock_data_dict, market_data
    
    def _extract_market_data_from_df(
        self,
        df: pd.DataFrame,
        market_column: str,
        risk_free_column: str
    ) -> MarketData:
        """Extract MarketData from DataFrame.
        
        Args:
            df: DataFrame with market data
            market_column: Market index column name
            risk_free_column: Risk-free rate column name
            
        Returns:
            MarketData instance
        """
        if market_column not in df.columns:
            raise DataLoaderError(f"Market column '{market_column}' not found")
        
        if risk_free_column not in df.columns:
            raise DataLoaderError(f"Risk-free rate column '{risk_free_column}' not found")
        
        # Extract market prices
        prices = df[market_column].copy()
        prices = prices.dropna()
        
        # Calculate returns
        returns = np.log(prices / prices.shift(1))
        
        # Calculate mean risk-free rate
        risk_free_series = df[risk_free_column].dropna()
        risk_free_mean = risk_free_series.mean()
        
        # Convert percentage to decimal if needed
        if risk_free_mean > 1:
            risk_free_rate = risk_free_mean / 100.0
        else:
            risk_free_rate = risk_free_mean
        
        return MarketData(
            index_name='Market Index',
            prices=prices,
            returns=returns,
            risk_free_rate=risk_free_rate
        )
    
    def _extract_stock_data_from_df(
        self,
        df: pd.DataFrame,
        ticker_columns: List[str],
        source_file: Union[str, Path]
    ) -> Dict[str, StockData]:
        """Extract StockData instances from DataFrame.
        
        Args:
            df: DataFrame with stock data
            ticker_columns: List of stock ticker column names
            source_file: Path to source file
            
        Returns:
            Dictionary mapping tickers to StockData instances
        """
        stock_data_dict = {}
        
        for ticker in ticker_columns:
            # Extract prices
            prices = df[ticker].copy()
            prices = prices.dropna()
            
            if len(prices) == 0:
                continue  # Skip tickers with no data
            
            # Calculate returns
            returns = np.log(prices / prices.shift(1))
            
            # Create StockData instance
            try:
                stock_data = StockData(
                    ticker=ticker,
                    prices=prices,
                    returns=returns,
                    metadata={
                        'source_file': str(source_file),
                        'num_original_rows': len(df),
                        'num_valid_prices': len(prices)
                    }
                )
                stock_data_dict[ticker] = stock_data
            except ValueError:
                # Skip stocks that fail validation
                continue
        
        return stock_data_dict
