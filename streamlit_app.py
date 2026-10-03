"""Entry point:  streamlit run streamlit_app.py   (also the Streamlit Cloud default).

Runs frontend/app.py, which registers the pages in frontend/pages via st.navigation.
Keeping the entry script outside frontend/ stops Streamlit from auto-detecting
frontend/pages as a legacy multipage folder."""
import os
import runpy
from pathlib import Path

# On Streamlit Community Cloud, settings come from the app's Secrets (TOML). Copy top-level
# string secrets into environment variables before the app's config is imported.
try:
    import streamlit as st
    for _k, _v in st.secrets.items():
        if isinstance(_v, (str, int, float)) and _k not in os.environ:
            os.environ[_k] = str(_v)
except Exception:   # no secrets file locally -> .env / environment variables are used
    pass

runpy.run_path(str(Path(__file__).resolve().parent / "frontend" / "app.py"), run_name="__main__")
