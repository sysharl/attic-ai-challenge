import streamlit as st
from interfaces.streamlit_app import main

if __name__ == "__main__":
    st.set_page_config(page_title="ATTIC AI", layout="wide")
    main()