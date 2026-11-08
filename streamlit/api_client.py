"""Semua komunikasi ke Flask API ada di sini (dipakai semua halaman)."""
import os

import requests
import streamlit as st

# Melewati halaman peringatan ngrok (free plan); tidak berpengaruh di Vercel/lokal.
HEADERS = {"ngrok-skip-browser-warning": "true"}
TIMEOUT = 30


def default_api_url() -> str:
    try:
        return st.secrets["API_URL"]
    except Exception:  # secrets.toml tidak ada
        return os.getenv("API_URL", "http://127.0.0.1:5000")


def health(api_url: str) -> dict:
    r = requests.get(f"{api_url}/health", headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=300, show_spinner="Menghubungi API...")
def fetch_schema(api_url: str) -> dict:
    r = requests.get(f"{api_url}/schema", headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def predict(api_url: str, payload: dict) -> dict:
    r = requests.post(f"{api_url}/predict", json=payload,
                      headers=HEADERS, timeout=TIMEOUT)
    if r.status_code == 400:
        raise ValueError("; ".join(r.json().get("details", [])))
    r.raise_for_status()
    return r.json()
