"""
Print Utilities Module

This module provides standardized print formatting functions for the OptimML Framework.
It ensures consistent and visually appealing console output throughout the application.
"""

import time
from datetime import datetime
import sys
from typing import Optional, List, Dict, Any

# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_section_header(title: str) -> None:
    """
    Print a formatted section header.
    
    Args:
        title (str): The title of the section
    """
    width = 80
    print("\n" + "=" * width)
    print(f"{Colors.BOLD}{Colors.BLUE}{title.center(width)}{Colors.ENDC}")
    print("=" * width)

def print_subsection_header(title: str) -> None:
    """
    Print a formatted subsection header.
    
    Args:
        title (str): The title of the subsection
    """
    width = 80
    print("\n" + "-" * width)
    print(f"{Colors.BOLD}{Colors.CYAN}{title}{Colors.ENDC}")
    print("-" * width)

def print_info(message: str) -> None:
    """
    Print an informational message.
    
    Args:
        message (str): The message to print
    """
    print(f"{Colors.BLUE}[INFO]{Colors.ENDC} {message}")

def print_success(message: str) -> None:
    """
    Print a success message.
    
    Args:
        message (str): The message to print
    """
    print(f"{Colors.GREEN}[SUCCESS]{Colors.ENDC} {message}")

def print_warning(message: str) -> None:
    """
    Print a warning message.
    
    Args:
        message (str): The message to print
    """
    print(f"{Colors.YELLOW}[WARNING]{Colors.ENDC} {message}")

def print_error(message: str) -> None:
    """
    Print an error message.
    
    Args:
        message (str): The message to print
    """
    print(f"{Colors.RED}[ERROR]{Colors.ENDC} {message}")

def print_progress(current: int, total: int, message: str = "") -> None:
    """
    Print a progress bar.
    
    Args:
        current (int): Current progress value
        total (int): Total value representing 100% completion
        message (str, optional): Additional message to display
    """
    bar_length = 40
    progress = float(current) / float(total)
    arrow = '█' * int(round(progress * bar_length))
    spaces = ' ' * (bar_length - len(arrow))
    
    sys.stdout.write(f"\r[{arrow}{spaces}] {int(progress * 100)}% {message}")
    sys.stdout.flush()
    
    if current == total:
        print()

def print_timestamp(message: str) -> None:
    """
    Print a message with a timestamp.
    
    Args:
        message (str): The message to print
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{Colors.BOLD}[{timestamp}]{Colors.ENDC} {message}")

def print_model_training_start(model_name: str, method: str) -> None:
    """
    Print a formatted message indicating the start of model training.
    
    Args:
        model_name (str): Name of the model being trained
        method (str): Training method (e.g., 'genetic', 'exhaustive')
    """
    print_timestamp(f"{Colors.BOLD}{Colors.CYAN}Starting training: {model_name}{Colors.ENDC} using {method} method...")

def print_model_training_complete(model_name: str, method: str, duration: float, metrics: Optional[Dict[str, Any]] = None) -> None:
    """
    Print a formatted message indicating the completion of model training.
    
    Args:
        model_name (str): Name of the model that was trained
        method (str): Training method used (e.g., 'genetic', 'exhaustive')
        duration (float): Training duration in seconds
        metrics (dict, optional): Dictionary of performance metrics
    """
    metrics_str = ""
    if metrics:
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        metrics_str = f" | {metrics_str}"
    
    print_timestamp(f"{Colors.BOLD}{Colors.GREEN}Completed training: {model_name}{Colors.ENDC} ({method}) in {duration:.2f}s{metrics_str}")

def print_data_loading(file_name: str, index: int, total: int) -> None:
    """
    Print a formatted message for data loading progress.
    
    Args:
        file_name (str): Name of the file being loaded
        index (int): Current file index
        total (int): Total number of files
    """
    print(f"{Colors.BLUE}[{index}/{total}]{Colors.ENDC} Loading dataset: {Colors.BOLD}{file_name}{Colors.ENDC}")

def print_summary_statistics(title: str, data: Dict[str, Any]) -> None:
    """
    Print a formatted summary of statistics.
    
    Args:
        title (str): Title for the statistics summary
        data (dict): Dictionary of statistics to display
    """
    print_subsection_header(title)
    max_key_length = max(len(str(k)) for k in data.keys())
    
    for key, value in data.items():
        key_str = str(key).ljust(max_key_length)
        print(f"  {Colors.BOLD}{key_str}{Colors.ENDC}: {value}")

def print_table(headers: List[str], rows: List[List[Any]]) -> None:
    """
    Print a formatted table.
    
    Args:
        headers (list): List of column headers
        rows (list): List of rows, where each row is a list of values
    """
    # Calculate column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))
    
    # Print header
    header_row = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    print(f"{Colors.BOLD}{header_row}{Colors.ENDC}")
    print("-" * len(header_row))
    
    # Print rows
    for row in rows:
        row_str = " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row))
        print(row_str)

def print_application_header() -> None:
    """
    Print the application header with version and copyright information.
    """
    width = 80
    print("\n" + "=" * width)
    print(f"{Colors.BOLD}{Colors.GREEN}OptimML Framework v1.0.0{Colors.ENDC}".center(width))
    print(f"An Optimized Machine Learning Framework".center(width))
    print(f"© {datetime.now().year} - All Rights Reserved".center(width))
    print("=" * width + "\n")
