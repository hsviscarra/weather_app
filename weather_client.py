"""
Client for the National Weather Service (NWS) API - api.weather.gov - plus
geocoding via Nominatim (OpenStreetMap), since NWS itself only accepts
lat/lon, not place names.

Both are free and keyless (unlike massive_client.py's Massive API), but both
require a descriptive User-Agent header identifying the app + contact info,
or requests may be rate-limited/blocked. No secret scope needed for either -
only Lakebase needs one (see lakebase.py / setup_secrets.py).
"""

import os
from typing import Any

import requests

_BASE_URL = os.environ.get("NWS_API_BASE_URL", "https://api.weather.gov")
_USER_AGENT = os.environ.get("NWS_USER_AGENT", "(weather_app, contact@example.com)")
_GEOCODE_URL = os.environ.get("GEOCODE_API_BASE_URL", "https://nominatim.openstreetmap.org/search")

_DEFAULT_TIMEOUT = 30


class WeatherClient:
    """Thin wrapper around the NWS API for alerts + forecasts."""

    def __init__(self, base_url: str | None = None, timeout: int = _DEFAULT_TIMEOUT):
        self.base_url = (base_url or _BASE_URL).rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": _USER_AGENT,
                "Accept": "application/geo+json",
            }
        )

    def get(self, path_or_url: str, params: dict[str, Any] | None = None) -> Any:
        url = path_or_url if path_or_url.startswith("http") else f"{self.base_url}{path_or_url}"
        resp = self._session.get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def get_active_alerts(self, lat: float, lon: float) -> list[dict]:
        """Active alerts affecting a specific point, in a SINGLE API call
        (GET /alerts/active?point={lat},{lon}). Returns the GeoJSON
        "features" list - each item's "properties" has: id, event, headline,
        description, instruction, severity, sent, effective, onset, expires.
        """
        data = self.get("/alerts/active", params={"point": f"{lat},{lon}"})
        return data.get("features", [])

    def get_forecast_periods(self, lat: float, lon: float) -> list[dict]:
        """Forecast periods for a specific point. Two API calls: first
        resolves the point to its forecast grid endpoint
        (GET /points/{lat},{lon}), then fetches the periods from that
        endpoint's "forecast" URL. Returns the "properties.periods" list -
        each item has: name, startTime, endTime, temperature, shortForecast,
        detailedForecast.
        """
        point = self.get(f"/points/{lat},{lon}")
        forecast_url = point["properties"]["forecast"]
        data = self.get(forecast_url)
        return data["properties"]["periods"]

    def geocode(self, place_name: str) -> tuple[float, float]:
        """Resolve a place name (e.g. "Chicago, IL") to (lat, lon) via the
        free, keyless Nominatim (OpenStreetMap) geocoder. Raises ValueError
        if no match is found. Nominatim's usage policy caps requests at
        1/sec - callers syncing multiple locations should pace calls (see
        app.py's POST /weather/sync)."""
        resp = self._session.get(
            _GEOCODE_URL,
            params={"q": place_name, "format": "json", "limit": 1},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        results = resp.json()
        if not results:
            raise ValueError(f"Could not geocode location: {place_name!r}")
        return float(results[0]["lat"]), float(results[0]["lon"])
