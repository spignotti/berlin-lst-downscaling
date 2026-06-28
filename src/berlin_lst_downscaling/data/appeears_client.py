"""NASA AppEEARS REST API client for programmatic data download.

Provides token-based authentication via Earthdata Login and a minimal
interface for submitting area extraction tasks, polling their status,
and downloading results.

Usage::

    client = AppEEARSClient()
    client.login(os.environ["EARTHDATA_USERNAME"], os.environ["EARTHDATA_PASSWORD"])
    task_id = client.submit_area_task(name="ecostress-berlin-2023", ...)
    client.wait_for_task(task_id)
    files = client.list_bundle_files(task_id)
    for f in files:
        client.download_file(task_id, f["file_id"], "/tmp/out.tif")
    client.logout()
"""

from __future__ import annotations

import logging
import os
import time
from hashlib import sha256
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

# ── Exceptions ────────────────────────────────────────────────────────────────


class AppEEARSError(Exception):
    """Generic AppEEARS API error."""


class AuthError(AppEEARSError):
    """Authentication failure (invalid credentials or expired token)."""


class TaskError(AppEEARSError):
    """Task submission or processing error."""


# ── Constants ─────────────────────────────────────────────────────────────────

_POLL_INTERVAL_DEFAULT = 60
_TIMEOUT_HOURS_DEFAULT = 24


# ── Client ────────────────────────────────────────────────────────────────────


class AppEEARSClient:
    """Thin wrapper around the AppEEARS REST API.

    Each instance holds an auth token obtained via Earthdata Login.
    The token auto-refreshes when a 403 is received.
    """

    def __init__(self, base_url: str = "https://appeears.earthdatacloud.nasa.gov/api") -> None:
        self.base_url = base_url.rstrip("/")
        self._token: str | None = None
        self._session = requests.Session()
        # Default headers for all requests
        self._session.headers.update(
            {"Content-Type": "application/json;charset=UTF-8"}
        )

    # ── Auth ───────────────────────────────────────────────────────────────

    def login(self, username: str, password: str) -> str:
        """Authenticate via Earthdata Login and cache the Bearer token.

        Args:
            username: NASA Earthdata Login username.
            password: NASA Earthdata Login password.

        Returns:
            The Bearer token string.

        Raises:
            AuthError: If credentials are invalid.
        """
        resp = self._session.post(
            f"{self.base_url}/login",
            auth=(username, password),
            # Empty body required by AppEEARS
            data="",
            headers={"Content-Type": "application/x-www-form-urlencoded;charset=UTF-8"},
        )
        if resp.status_code == 401:
            raise AuthError("Invalid Earthdata credentials (HTTP 401).")
        if not resp.ok:
            raise AppEEARSError(
                f"AppEEARS login failed with status {resp.status_code}"
            )
        payload = resp.json()
        self._token = str(payload["token"])
        logger.info("AppEEARS login successful (token expires ~48h).")
        return self._token

    def logout(self) -> None:
        """Invalidate the current token.

        Best-effort; does not raise on failure.
        """
        if not self._token:
            return
        try:
            self._session.post(
                f"{self.base_url}/logout",
                headers=self._auth_header(),
            )
        except requests.RequestException as exc:
            logger.warning("AppEEARS logout failed: %s", exc)
        finally:
            self._token = None

    def _auth_header(self) -> dict[str, str]:
        """Return the Authorization header dict for the cached token.

        Raises:
            AuthError: If not logged in.
        """
        if self._token is None:
            raise AuthError("Not logged in. Call login() first.")
        return {"Authorization": f"Bearer {self._token}"}

    def _ensure_auth(self) -> None:
        """Re-authenticate if the cached token is missing."""
        if self._token is None:
            username = os.environ.get("EARTHDATA_USERNAME", "")
            password = os.environ.get("EARTHDATA_PASSWORD", "")
            if not username or not password:
                raise AuthError(
                    "EARTHDATA_USERNAME and EARTHDATA_PASSWORD must be set in the environment. "
                    "See .env.example for details."
                )
            self.login(username, password)

    def _request(
        self,
        method: str,
        path: str,
        *,
        auto_auth: bool = True,
        **kwargs: Any,
    ) -> requests.Response:
        """Make an API request with automatic auth and token refresh.

        Args:
            method: HTTP method.
            path: URL path relative to ``base_url``.
            auto_auth: If True, add Authorization header and retry on 403.
            **kwargs: Passed to ``requests.Session.request``.

        Returns:
            Response with status 2xx.

        Raises:
            AuthError: If auth fails even after refresh.
            AppEEARSError: On HTTP error.
        """
        if auto_auth:
            self._ensure_auth()
            headers = kwargs.pop("headers", {})
            headers.update(self._auth_header())
            kwargs["headers"] = headers

        url = f"{self.base_url}{path}"
        # Ensure a reasonable timeout on every request
        kwargs.setdefault("timeout", (30, 600))  # 30s connect, 10min read
        resp = self._session.request(method, url, **kwargs)

        # Auto-refresh on 403
        if resp.status_code == 403 and auto_auth:
            logger.info("Token expired; re-authenticating...")
            self._token = None
            self._ensure_auth()
            headers = kwargs["headers"]
            headers.update(self._auth_header())
            kwargs.setdefault("timeout", (30, 600))
            resp = self._session.request(method, url, **kwargs)

        if not resp.ok:
            detail = resp.text[:500] if resp.text else "(no body)"
            raise AppEEARSError(
                f"AppEEARS {method} {path} returned {resp.status_code}: {detail}"
            )
        return resp

    # ── Tasks ──────────────────────────────────────────────────────────────

    def submit_area_task(
        self,
        name: str,
        geo_json: dict[str, Any],
        layers: list[dict[str, str]],
        dates: list[dict[str, Any]],
        output_format: str = "geotiff",
        projection: str = "native",
        filename_date: str | None = "calendar",
    ) -> str:
        """Submit an area extraction task.

        Args:
            name: Human-readable task name.
            geo_json: GeoJSON FeatureCollection defining the AOI (EPSG:4326).
            layers: List of ``{"product": "...", "layer": "..."}`` dicts.
            dates: List of date specification dicts (``startDate``, ``endDate``,
                   ``recurring``, ``yearRange``).
            output_format: ``"geotiff"`` or ``"netcdf4"``.
            projection: Output projection (``"native"``, ``"geographic"``, etc.).
            filename_date: ``"calendar"`` for calendar dates in filenames,
                           or ``None`` for DOY format.

        Returns:
            The task ID string.

        Raises:
            TaskError: If the task submission fails.
        """
        # Build output config
        output_fmt: dict[str, Any] = {
            "type": output_format,
        }
        if filename_date:
            output_fmt["filename_date"] = filename_date

        payload: dict[str, Any] = {
            "task_type": "area",
            "task_name": name,
            "params": {
                "dates": dates,
                "layers": layers,
                "geo": geo_json,
                "output": {
                    "format": output_fmt,
                    "projection": projection,
                },
            },
        }

        resp = self._request("POST", "/task", json=payload)
        result: dict[str, Any] = resp.json()
        task_id: str | None = result.get("task_id")
        if not task_id:
            raise TaskError(f"Task submission did not return a task_id: {result}")
        logger.info("AppEEARS task submitted: %s (%s)", name, task_id)
        return task_id

    def get_task_status(self, task_id: str) -> dict[str, Any]:
        """Get the current status of a task.

        Returns a dict with at least ``"status"`` (one of ``pending``,
        ``processing``, ``done``, ``error``).
        """
        resp = self._request("GET", f"/task/{task_id}")
        return resp.json()

    def wait_for_task(
        self,
        task_id: str,
        poll_interval_sec: int = _POLL_INTERVAL_DEFAULT,
        timeout_hours: int = _TIMEOUT_HOURS_DEFAULT,
    ) -> dict[str, Any]:
        """Poll a task until it completes or fails.

        Args:
            task_id: Task to wait for.
            poll_interval_sec: Seconds between status checks.
            timeout_hours: Maximum wall-clock wait time.

        Returns:
            The final task status dict.

        Raises:
            TaskError: If the task fails or times out.
        """
        deadline = time.time() + timeout_hours * 3600

        while time.time() < deadline:
            status = self.get_task_status(task_id)
            state = status.get("status", "unknown")

            if state == "done":
                logger.info("AppEEARS task %s completed.", task_id)
                # After completion, the endpoint returns a 303 with Location
                # pointing to /task/{task_id}. Our _request follows this.
                return status
            if state in ("error", "failed"):
                err = status.get("error", "Unknown error")
                raise TaskError(f"AppEEARS task {task_id} failed: {err}")
            if state == "processing":
                # Optionally fetch detailed progress
                progress_response = self._request("GET", f"/status/{task_id}").json()
                detail = "?"
                if isinstance(progress_response, list) and progress_response:
                    detail = progress_response[0].get("progress", {}).get("summary", "?")
                logger.info("  %s: %s%% complete", task_id, detail)
            else:
                logger.info("  %s: state=%s", task_id, state)

            time.sleep(poll_interval_sec)

        raise TaskError(
            f"AppEEARS task {task_id} timed out after {timeout_hours}h."
        )

    # ── Bundle / Download ──────────────────────────────────────────────────

    def list_bundle_files(self, task_id: str) -> list[dict[str, Any]]:
        """List files in a completed task's bundle.

        Returns a list of dicts with keys ``file_id``, ``file_name``,
        ``sha256``, etc.
        """
        resp = self._request("GET", f"/bundle/{task_id}")
        bundle: dict[str, Any] = resp.json()
        return list(bundle.get("files", []))

    def download_file(
        self,
        task_id: str,
        file_id: str,
        dest_path: str | Path,
        *,
        verify_sha256: bool = True,
    ) -> Path:
        """Download a single file from a completed task's bundle.

        Uses streaming download to avoid large memory buffers.

        Args:
            task_id: The completed task ID.
            file_id: The file ID from ``list_bundle_files``.
            dest_path: Local path to save the file.
            verify_sha256: If True, verify the file checksum.

        Returns:
            The resolved ``Path`` to the downloaded file.

        Raises:
            AppEEARSError: If the download or verify fails.
        """
        dest = Path(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        # Use GET /bundle/{task_id}/{file_id} for download
        url = f"{self.base_url}/bundle/{task_id}/{file_id}"
        self._ensure_auth()
        headers = self._auth_header()
        # Stream download
        with self._session.get(url, headers=headers, stream=True) as resp:
            if not resp.ok:
                # Try auto-refresh
                self._token = None
                self._ensure_auth()
                headers = self._auth_header()
                resp = self._session.get(url, headers=headers, stream=True)
            resp.raise_for_status()

            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):  # 8MB
                    f.write(chunk)

        if verify_sha256:
            self._verify_checksum(dest, file_id, task_id)

        logger.info("Downloaded: %s", dest)
        return dest

    def _verify_checksum(self, path: Path, file_id: str, task_id: str) -> None:
        """Verify a downloaded file's SHA-256 against the bundle metadata.

        Raises:
            AppEEARSError: On checksum mismatch.
        """
        # Fetch bundle entry for this file_id to get the known hash
        bundle = self.list_bundle_files(task_id)
        known_hash: str | None = None
        for f_entry in bundle:
            if f_entry.get("file_id") == file_id:
                known_hash = f_entry.get("sha256")
                break

        if not known_hash:
            logger.warning("No SHA-256 in bundle metadata for %s; skipping verify.", file_id)
            return

        computed = sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                computed.update(chunk)

        if computed.hexdigest() != known_hash:
            path.unlink(missing_ok=True)
            raise AppEEARSError(
                f"SHA-256 mismatch for {path.name}: "
                f"computed={computed.hexdigest()}, expected={known_hash}"
            )

    # ── Product info ───────────────────────────────────────────────────────

    def list_products(self) -> list[dict[str, Any]]:
        """List all available AppEEARS products."""
        resp = self._request("GET", "/product")
        return list(resp.json())

    def get_product_layers(self, product_id: str) -> dict[str, Any]:
        """Get layer information for a specific product.

        Example: ``get_product_layers("ECO_L2T_LSTE.002")``
        """
        resp = self._request("GET", f"/product/{product_id}")
        return resp.json()


# ── Convenience factory ───────────────────────────────────────────────────────


def appeears_client_from_env(
    base_url: str = "https://appeears.earthdatacloud.nasa.gov/api",
) -> AppEEARSClient:
    """Create and pre-authenticate an AppEEARS client from environment variables.

    Requires ``EARTHDATA_USERNAME`` and ``EARTHDATA_PASSWORD`` in the environment.
    """
    client = AppEEARSClient(base_url=base_url)
    client._ensure_auth()
    return client
