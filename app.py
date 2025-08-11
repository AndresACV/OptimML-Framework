import streamlit as st

# Configure Streamlit page - MUST be the first Streamlit command
st.set_page_config(
    page_title="OptimML Framework Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

import pandas as pd
import numpy as np
import warnings
import os
import sys

"""
OptimML Framework - Main Application Entry Point

This file serves as the entry point for the OptimML Framework application.
It configures the environment, sets up the Streamlit interface, and launches
the main application functionality from the components package.
"""

# Add the components directory to the path so we can import modules
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "components"))

# Import the main application module
from components import CMain

# Configure pandas to avoid Arrow serialization issues
pd.options.mode.copy_on_write = True
pd.options.future.infer_string = False

# Additional settings to help with Arrow compatibility
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Suppress common warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

def main():
    """
    Main application function that launches the application.
    
    This function calls the main function from the CMain module,
    which handles the core application logic.
    
    Exceptions are caught and displayed to the user in a friendly format.
    """
    # Call the main function from CMain
    try:
        CMain.main()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.exception(e)

if __name__ == "__main__":
    main()
