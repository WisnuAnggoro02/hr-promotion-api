"""Menguji API (bukan melatih). Dua mode:

    python scripts/collection.py                 # tes lokal via Flask test client
    python scripts/collection.py --url https://hr-promotion-api.vercel.app
"""
import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

HIGH = {"department": "Analytics", "region": "region_2", "education": "Master's & above",
        "gender": "Male", "recruitment_channel": "Sourcing", "no_of_trainings": 1,
        "age": 30, "previous_year_rating": 5, "length_of_service": 4,
        "awards_won": 1, "avg_training_score": 92}
LOW = {**HIGH, "previous_year_rating": 1, "awards_won": 0, "avg_training_score": 45}
BAD = {**HIGH, "gender": "Robot", "age": 150}
MISSING = {k: v for k, v in HIGH.items() if k != "age"}

CASES = [  # (nama, method, path, body, status yang diharapkan)
    ("health", "GET", "/health", None, 200),
    ("schema", "GET", "/schema", None, 200),
    ("profil kuat", "POST", "/predict", HIGH, 200),
    ("profil lemah", "POST", "/predict", LOW, 200),
    ("input tidak valid", "POST", "/predict", BAD, 400),
    ("field hilang", "POST", "/predict", MISSING, 400),
]


def make_caller(base_url):
    if base_url:
        def call(method, path, body):
            data = json.dumps(body).encode() if body is not None else None
            req = urllib.request.Request(
                base_url.rstrip("/") + path, data=data, method=method,
                headers={"Content-Type": "application/json",
                         "ngrok-skip-browser-warning": "true"})
            try:
                with urllib.request.urlopen(req, timeout=30) as r:
                    return r.status, json.loads(r.read())
            except urllib.error.HTTPError as e:
                return e.code, json.loads(e.read())
        return call

    sys.path.insert(0, str(ROOT))
    from api.index import app
    client = app.test_client()

    def call(method, path, body):
        r = client.open(path, method=method, json=body)
        return r.status_code, r.get_json()
    return call


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", help="URL API (kosong = tes lokal tanpa server)")
    call = make_caller(parser.parse_args().url)

    failed = 0
    for name, method, path, body, expected in CASES:
        status, data = call(method, path, body)
        ok = status == expected
        failed += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name:<18} {method:<4} {path:<9} "
              f"-> {status} {json.dumps(data)[:90]}")
    print(f"\n{len(CASES) - failed}/{len(CASES)} tes lulus")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
