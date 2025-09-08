"""
Thin runner for the Flask web application.

This file intentionally has no CLI — it exposes `app` so `flask run` works,
and allows `python app.py` to start the server for local use.
"""

from webapp import app  # noqa: F401

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080)
