import os
import sys
import importlib.util

# Standard cloud entrypoint for Streamlit deployment
file_path = os.path.join(os.path.dirname(__file__), "web app.py")
spec = importlib.util.spec_from_file_location("web_app", file_path)
web_app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(web_app)

if __name__ == '__main__':
    web_app.main()
