import sys
from src.logger import logging


def error_message_detail(error, error_detail: sys):
    """
    Generate detailed error message with file name and line number.

    Args:
        error: The exception object
        error_detail: System information about the exception

    Returns:
        Formatted error message with details
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno

    error_message = f"Error occurred in python script [{file_name}] at line number [{line_number}] - error message: [{str(error)}]"
    return error_message


class CustomException(Exception):
    """
    Custom exception class for handling errors in the application.
    Includes logging functionality to track errors.
    """
    
    def __init__(self, error_message, error_detail: sys):
        """
        Initialize the custom exception
        
        Args:
            error_message: The error message to display
            error_detail: System information about the error
        """
        super().__init__(error_message)
        self.error_message = self._get_error_message(error_message, error_detail)
        
    def _get_error_message(self, error_message, error_detail: sys) -> str:
        """
        Format the error message with file and line number information
        
        Args:
            error_message: The original error message
            error_detail: System information about the error
            
        Returns:
            Formatted error message string
        """
        _, _, exc_tb = error_detail.exc_info()
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
        
        error_message = f"Error occurred in python script [{file_name}] line number [{line_number}] error message [{error_message}]"
        
        logging.error(error_message)
        return error_message
    
    def __str__(self):
        """String representation of the error"""
        return self.error_message
