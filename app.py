import streamlit as st
import pandas as pd
import numpy as np
import warnings
import os
import sys

# Add the current directory to the path so we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the main application module
import CMain

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

# Create a Streamlit app wrapper
def main():
    st.set_page_config(
        page_title="Framework Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # Call the main function from CMain
    try:
        CMain.main()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.exception(e)

if __name__ == "__main__":
    main()
