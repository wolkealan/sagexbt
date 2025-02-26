from typing import Dict, List, Any, Optional, Union
import re
from datetime import datetime

from config.config import TradingConfig
from utils.logger import get_logger

logger = get_logger("validators")

class Validators:
    """Provides validation utilities for input data"""
    
    @staticmethod
    def validate_symbol(symbol: str) -> bool:
        """
        Validate cryptocurrency symbol
        
        Args:
            symbol: Cryptocurrency symbol to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            if not symbol:
                return False
            
            # Convert to uppercase
            symbol = symbol.upper()
            
            # Check if symbol is in supported coins list
            return symbol in TradingConfig.SUPPORTED_COINS
        
        except Exception as e:
            logger.error(f"Error validating symbol: {e}")
            return False
    
    @staticmethod
    def validate_timeframe(timeframe: str) -> bool:
        """
        Validate timeframe string
        
        Args:
            timeframe: Timeframe string to validate (e.g., "1d", "4h")
            
        Returns:
            True if valid, False otherwise
        """
        try:
            if not timeframe:
                return False
            
            # Check format using regex (number followed by one of m, h, d, w)
            pattern = r"^[1-9][0-9]*[mhdw]$"
            return bool(re.match(pattern, timeframe))
        
        except Exception as e:
            logger.error(f"Error validating timeframe: {e}")
            return False
    
    @staticmethod
    def validate_date(date_str: str) -> Optional[datetime]:
        """
        Validate and parse date string
        
        Args:
            date_str: Date string to validate
            
        Returns:
            Datetime object if valid, None otherwise
        """
        try:
            # Try parsing with multiple formats
            formats = [
                "%Y-%m-%d",        # 2023-01-31
                "%Y-%m-%dT%H:%M:%S",  # 2023-01-31T12:34:56
                "%Y-%m-%dT%H:%M:%SZ", # 2023-01-31T12:34:56Z
                "%d/%m/%Y",        # 31/01/2023
                "%m/%d/%Y",        # 01/31/2023
                "%Y/%m/%d",        # 2023/01/31
                "%d-%m-%Y",        # 31-01-2023
                "%m-%d-%Y",        # 01-31-2023
                "%b %d, %Y",       # Jan 31, 2023
                "%d %b %Y",        # 31 Jan 2023
                "%B %d, %Y",       # January 31, 2023
                "%d %B %Y"         # 31 January 2023
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue
            
            # No format matched
            return None
        
        except Exception as e:
            logger.error(f"Error validating date: {e}")
            return None
    
    @staticmethod
    def validate_price(price: Union[str, float]) -> Optional[float]:
        """
        Validate and parse price value
        
        Args:
            price: Price value to validate
            
        Returns:
            Float price if valid, None otherwise
        """
        try:
            # If already a float, check if positive
            if isinstance(price, (int, float)):
                return float(price) if price >= 0 else None
            
            # Try to convert string to float
            if isinstance(price, str):
                # Remove currency symbols and commas
                price = price.replace('$', '').replace('€', '').replace('£', '').replace(',', '')
                price_float = float(price)
                return price_float if price_float >= 0 else None
            
            return None
        
        except Exception as e:
            logger.error(f"Error validating price: {e}")
            return None
    
    @staticmethod
    def validate_percentage(percentage: Union[str, float]) -> Optional[float]:
        """
        Validate and parse percentage value
        
        Args:
            percentage: Percentage value to validate
            
        Returns:
            Float percentage if valid, None otherwise
        """
        try:
            # If already a float, check if in reasonable range
            if isinstance(percentage, (int, float)):
                return float(percentage) if -100 <= percentage <= 1000 else None
            
            # Try to convert string to float
            if isinstance(percentage, str):
                # Remove percentage symbol
                percentage = percentage.replace('%', '')
                percentage_float = float(percentage)
                return percentage_float if -100 <= percentage_float <= 1000 else None
            
            return None
        
        except Exception as e:
            logger.error(f"Error validating percentage: {e}")
            return None
    
    @staticmethod
    def validate_action_type(action_type: str) -> bool:
        """
        Validate trading action type
        
        Args:
            action_type: Action type to validate (e.g., "spot", "futures")
            
        Returns:
            True if valid, False otherwise
        """
        try:
            if not action_type:
                return False
            
            # Convert to lowercase
            action_type = action_type.lower()
            
            # Check if action type is supported
            return action_type in ["spot", "futures"]
        
        except Exception as e:
            logger.error(f"Error validating action type: {e}")
            return False
    
    @staticmethod
    def validate_risk_tolerance(risk_tolerance: str) -> bool:
        """
        Validate risk tolerance level
        
        Args:
            risk_tolerance: Risk tolerance to validate (e.g., "low", "medium", "high")
            
        Returns:
            True if valid, False otherwise
        """
        try:
            if not risk_tolerance:
                return False
            
            # Convert to lowercase
            risk_tolerance = risk_tolerance.lower()
            
            # Check if risk tolerance is supported
            return risk_tolerance in ["low", "medium", "high"]
        
        except Exception as e:
            logger.error(f"Error validating risk tolerance: {e}")
            return False
    
    @staticmethod
    def validate_mongo_id(mongo_id: str) -> bool:
        """
        Validate MongoDB ObjectId string
        
        Args:
            mongo_id: MongoDB ObjectId string to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            if not mongo_id:
                return False
            
            # MongoDB ObjectId is a 24-character hex string
            pattern = r"^[0-9a-fA-F]{24}$"
            return bool(re.match(pattern, mongo_id))
        
        except Exception as e:
            logger.error(f"Error validating MongoDB ObjectId: {e}")
            return False
    
    @staticmethod
    def validate_api_key(api_key: str) -> bool:
        """
        Validate API key format
        
        Args:
            api_key: API key string to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            if not api_key:
                return False
            
            # API keys are typically alphanumeric strings of reasonable length
            pattern = r"^[a-zA-Z0-9_\-]{8,128}$"
            return bool(re.match(pattern, api_key))
        
        except Exception as e:
            logger.error(f"Error validating API key: {e}")
            return False
    
    @staticmethod
    def validate_url(url: str) -> bool:
        """
        Validate URL format
        
        Args:
            url: URL string to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            if not url:
                return False
            
            # Basic URL pattern
            pattern = r"^(https?:\/\/)?(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)$"
            return bool(re.match(pattern, url))
        
        except Exception as e:
            logger.error(f"Error validating URL: {e}")
            return False
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """
        Validate email format
        
        Args:
            email: Email string to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            if not email:
                return False
            
            # Basic email pattern
            pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
            return bool(re.match(pattern, email))
        
        except Exception as e:
            logger.error(f"Error validating email: {e}")
            return False
    
    @staticmethod
    def validate_query_params(params: Dict[str, Any], required_params: List[str] = None,
                           optional_params: List[str] = None) -> Dict[str, Any]:
        """
        Validate query parameters
        
        Args:
            params: Parameter dictionary to validate
            required_params: List of required parameter names
            optional_params: List of optional parameter names
            
        Returns:
            Dictionary with validation results
        """
        try:
            validation_results = {
                "valid": True,
                "missing_required": [],
                "unknown_params": [],
                "validated_params": {}
            }
            
            # Check required parameters
            if required_params:
                for param in required_params:
                    if param not in params:
                        validation_results["valid"] = False
                        validation_results["missing_required"].append(param)
            
            # Check for unknown parameters
            all_known_params = set()
            if required_params:
                all_known_params.update(required_params)
            if optional_params:
                all_known_params.update(optional_params)
            
            for param in params:
                if param not in all_known_params:
                    validation_results["unknown_params"].append(param)
            
            # Validate and type-convert known parameters
            for param, value in params.items():
                if param not in all_known_params:
                    continue
                
                # Specific validation based on parameter name
                if param == "symbol":
                    if Validators.validate_symbol(value):
                        validation_results["validated_params"][param] = value.upper()
                    else:
                        validation_results["valid"] = False
                
                elif param == "timeframe":
                    if Validators.validate_timeframe(value):
                        validation_results["validated_params"][param] = value
                    else:
                        validation_results["valid"] = False
                
                elif param == "date" or param.endswith("_date"):
                    date_obj = Validators.validate_date(value)
                    if date_obj:
                        validation_results["validated_params"][param] = date_obj
                    else:
                        validation_results["valid"] = False
                
                elif param == "price" or param.endswith("_price"):
                    price_val = Validators.validate_price(value)
                    if price_val is not None:
                        validation_results["validated_params"][param] = price_val
                    else:
                        validation_results["valid"] = False
                
                elif param.startswith("pct") or param.endswith("_pct") or param.endswith("_percentage"):
                    pct_val = Validators.validate_percentage(value)
                    if pct_val is not None:
                        validation_results["validated_params"][param] = pct_val
                    else:
                        validation_results["valid"] = False
                
                elif param == "action_type":
                    if Validators.validate_action_type(value):
                        validation_results["validated_params"][param] = value.lower()
                    else:
                        validation_results["valid"] = False
                
                elif param == "risk_tolerance":
                    if Validators.validate_risk_tolerance(value):
                        validation_results["validated_params"][param] = value.lower()
                    else:
                        validation_results["valid"] = False
                
                else:
                    # For other parameters, just copy them as-is
                    validation_results["validated_params"][param] = value
            
            return validation_results
        
        except Exception as e:
            logger.error(f"Error validating query parameters: {e}")
            return {
                "valid": False,
                "error": str(e),
                "validated_params": {}
            }

# Example usage
if __name__ == "__main__":
    # Test symbol validation
    print(f"BTC is valid symbol: {Validators.validate_symbol('BTC')}")
    print(f"XYZ is valid symbol: {Validators.validate_symbol('XYZ')}")
    
    # Test timeframe validation
    print(f"1d is valid timeframe: {Validators.validate_timeframe('1d')}")
    print(f"1x is valid timeframe: {Validators.validate_timeframe('1x')}")
    
    # Test date validation
    print(f"Parse 2023-01-31: {Validators.validate_date('2023-01-31')}")
    print(f"Parse invalid date: {Validators.validate_date('2023-13-31')}")