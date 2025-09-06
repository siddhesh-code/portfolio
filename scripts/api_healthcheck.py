#!/usr/bin/env python3
"""
Parallel API Healthcheck for the Flask app.

Checks core endpoints concurrently and validates responses.

Usage examples:
  python scripts/api_healthcheck.py
  python scripts/api_healthcheck.py --base-url http://127.0.0.1:5000 --concurrency 6
  python scripts/api_healthcheck.py --iterations 3 --interval 2
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple, Callable

import requests


def _ok(cond: bool, msg: str = "") -> Tuple[bool, str]:
    return cond, msg


class APIHealthcheck:
    def __init__(self, base_url: str, timeout: float = 8.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def check_index_html(self) -> Tuple[str, bool, str]:
        name = "GET /"
        url = f"{self.base_url}/"
        try:
            r = requests.get(url, timeout=self.timeout)
            if r.status_code != 200:
                return name, False, f"status={r.status_code}"
            ctype = r.headers.get("Content-Type", "")
            ok, reason = _ok("text/html" in ctype, f"ctype={ctype}")
            return name, ok, reason
        except Exception as e:
            return name, False, str(e)

    def check_analyze_json(self, symbol: str = "RELIANCE.NS") -> Tuple[str, bool, str]:
        name = "POST /analyze (JSON)"
        url = f"{self.base_url}/analyze"
        try:
            r = requests.post(url, json={"symbol": symbol}, timeout=self.timeout)
            if r.status_code != 200:
                return name, False, f"status={r.status_code}"
            data = r.json()
            keys = ["price", "volume", "indicators"]
            missing = [k for k in keys if k not in data]
            if missing:
                return name, False, f"missing={missing}"
            return name, True, "ok"
        except Exception as e:
            return name, False, str(e)

    def check_api_analyze_bulk(self, symbols: List[str] | None = None) -> Tuple[str, bool, str]:
        name = "POST /api/analyze"
        url = f"{self.base_url}/api/analyze"
        payload = {"symbols": symbols or ["RELIANCE.NS", "TCS.NS"]}
        try:
            r = requests.post(url, json=payload, timeout=self.timeout)
            if r.status_code != 200:
                return name, False, f"status={r.status_code}"
            data = r.json()
            if not isinstance(data, dict) or "stocks" not in data:
                return name, False, "bad schema"
            stocks = data.get("stocks") or []
            if not isinstance(stocks, list) or len(stocks) == 0:
                return name, False, "no stocks returned"
            one = stocks[0]
            missing = [k for k in ["symbol", "indicators"] if k not in one]
            if missing:
                return name, False, f"missing={missing}"
            return name, True, f"ok ({len(stocks)} stocks)"
        except Exception as e:
            return name, False, str(e)

    def check_watchlist_flow(self) -> Tuple[str, bool, str]:
        name = "GET/POST /api/watchlist"
        try:
            # GET
            r = requests.get(f"{self.base_url}/api/watchlist", timeout=self.timeout)
            if r.status_code != 200:
                return name, False, f"GET status={r.status_code}"
            wl = r.json()
            if not isinstance(wl, list):
                return name, False, "GET not list"

            # POST add
            sym = "TESTSYM.NS"
            r2 = requests.post(f"{self.base_url}/api/watchlist", json={"symbol": sym, "action": "add"}, timeout=self.timeout)
            if r2.status_code != 200:
                return name, False, f"POST add status={r2.status_code}"
            # Confirm present (GET)
            r3 = requests.get(f"{self.base_url}/api/watchlist", timeout=self.timeout)
            if sym not in r3.json():
                return name, False, "symbol not added"

            # POST remove
            r4 = requests.post(f"{self.base_url}/api/watchlist", json={"symbol": sym, "action": "remove"}, timeout=self.timeout)
            if r4.status_code != 200:
                return name, False, f"POST remove status={r4.status_code}"
            return name, True, "ok"
        except Exception as e:
            return name, False, str(e)

    def check_alerts_flow(self) -> Tuple[str, bool, str]:
        name = "GET/POST /api/alerts"
        try:
            # GET
            r = requests.get(f"{self.base_url}/api/alerts", timeout=self.timeout)
            if r.status_code != 200:
                return name, False, f"GET status={r.status_code}"
            alerts = r.json()
            if not isinstance(alerts, list):
                return name, False, "GET not list"

            # POST valid alert
            payload = {"symbol": "RELTEST.NS", "condition": "above", "price": 123.45}
            r2 = requests.post(f"{self.base_url}/api/alerts", json=payload, timeout=self.timeout)
            if r2.status_code != 200:
                return name, False, f"POST status={r2.status_code}"
            data = r2.json()
            if not data.get("ok"):
                return name, False, "POST not ok"
            return name, True, "ok"
        except Exception as e:
            return name, False, str(e)


def run_once(hc: APIHealthcheck) -> List[Tuple[str, bool, str]]:
    checks: List[Callable[[], Tuple[str, bool, str]]] = [
        hc.check_index_html,
        hc.check_analyze_json,
        hc.check_api_analyze_bulk,
        hc.check_watchlist_flow,
        hc.check_alerts_flow,
    ]
    results: List[Tuple[str, bool, str]] = []
    with futures.ThreadPoolExecutor(max_workers=len(checks)) as ex:
        fs = [ex.submit(c) for c in checks]
        for f in futures.as_completed(fs):
            results.append(f.result())
    # Stable order
    results.sort(key=lambda x: x[0])
    return results


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Parallel API healthcheck")
    parser.add_argument("--base-url", default=os.getenv("BASE_URL", "http://127.0.0.1:8080"), help="Base URL of the running app")
    parser.add_argument("--timeout", type=float, default=float(os.getenv("HC_TIMEOUT", 8.0)), help="Per-request timeout in seconds")
    parser.add_argument("--iterations", type=int, default=1, help="Number of times to run all checks")
    parser.add_argument("--interval", type=float, default=0.0, help="Seconds to sleep between iterations")
    parser.add_argument("--json", action="store_true", help="Print JSON summary output")
    args = parser.parse_args(argv)

    hc = APIHealthcheck(args.base_url, timeout=args.timeout)
    overall_ok = True
    all_results: List[Dict[str, Any]] = []
    for i in range(args.iterations):
        if i > 0 and args.interval > 0:
            time.sleep(args.interval)
        results = run_once(hc)
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        if not args.json:
            print(f"\nIteration {i+1}: {ts}")
        for name, ok, info in results:
            overall_ok = overall_ok and ok
            all_results.append({"name": name, "ok": ok, "info": info, "time": ts})
            if not args.json:
                status = "PASS" if ok else "FAIL"
                print(f"[{status}] {name}: {info}")

    if args.json:
        print(json.dumps({"ok": overall_ok, "results": all_results}, indent=2))
    else:
        print(f"\nOverall: {'OK' if overall_ok else 'FAIL'}")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

