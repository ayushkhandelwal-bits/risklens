"""Entry point:  streamlit run streamlit_app.py   (also the Streamlit Cloud default).

Runs frontend/app.py, which registers the pages in frontend/pages via st.navigation.
Keeping the entry script outside frontend/ stops Streamlit from auto-detecting
frontend/pages as a legacy multipage folder."""
import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parent / "frontend" / "app.py"), run_name="__main__")
