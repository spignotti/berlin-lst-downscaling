"""Unit tests for the AppEEARS REST API client — error paths and retry logic only.

Most of the client's surface is a thin wrapper over ``requests.Session``; mock
tests against the HTTP layer test mock fidelity, not real behavior. We keep
only the tests that exercise non-trivial internal control flow:

  * Error-path tests for malformed API responses
  * The 403 → re-authenticate → retry loop (the only piece of real logic in
    the client besides status polling)
  * HTTP error propagation
  * Missing-credentials detection
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from berlin_lst_downscaling.data.appeears_client import (
    AppEEARSClient,
    AppEEARSError,
    AuthError,
    TaskError,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def client() -> AppEEARSClient:
    """Return an unauthenticated AppEEARSClient with a fake base URL."""
    return AppEEARSClient(base_url="https://appeears.test")


def _mock_response(
    status_code: int = 200,
    ok: bool = True,
    json_data: object = None,
    text: str = "",
) -> MagicMock:
    """Build a lightweight mock response."""
    m = MagicMock()
    m.status_code = status_code
    m.ok = ok
    m.text = text
    if json_data is not None:
        m.json.return_value = json_data
    return m


# ── Missing env credentials ───────────────────────────────────────────────────


def test_ensure_auth_missing_env(client: AppEEARSClient) -> None:
    """``_ensure_auth`` raises ``AuthError`` if env vars are missing."""
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(AuthError, match="EARTHDATA_USERNAME"):
            client._ensure_auth()


# ── Task submission error path ────────────────────────────────────────────────


def test_submit_area_task_no_task_id(client: AppEEARSClient) -> None:
    """Raises ``TaskError`` if response has no task_id."""
    client._token = "tok"

    mock_resp = _mock_response(200, True, {"status": "error"})

    with patch.object(client._session, "request", return_value=mock_resp):
        with pytest.raises(TaskError, match="did not return a task_id"):
            client.submit_area_task("test", {}, [], [])


# ── Task status error path ────────────────────────────────────────────────────


def test_wait_for_task_error(client: AppEEARSClient) -> None:
    """``wait_for_task`` raises ``TaskError`` on task error."""
    client._token = "tok"

    mock_resp = _mock_response(200, True, {"status": "error", "error": "download failed"})

    with patch.object(client._session, "request", return_value=mock_resp):
        with pytest.raises(TaskError, match="download failed"):
            client.wait_for_task("t-1", poll_interval_sec=1, timeout_hours=1)


# ── 403 → re-authenticate → retry loop ───────────────────────────────────────


def test_auto_reauth_on_403(client: AppEEARSClient) -> None:
    """``_request`` re-authenticates on 403 and retries the original request."""
    client._token = "expired"

    fail_resp = _mock_response(403, False, text="Forbidden")
    ok_resp = _mock_response(200, True, {"status": "ok"})
    login_resp = _mock_response(200, True, {"token": "new-token"})

    call_count = 0

    def mock_request(method: str, url: str, **kwargs: object) -> MagicMock:
        nonlocal call_count
        call_count += 1
        if "login" in url:
            return login_resp
        if call_count == 1:
            return fail_resp
        return ok_resp

    with patch.object(client._session, "request", side_effect=mock_request):
        env = {"EARTHDATA_USERNAME": "u", "EARTHDATA_PASSWORD": "p"}
        with patch.dict("os.environ", env):
            resp = client._request("GET", "/status/t-1")

    # Should have the 200 response from the retry
    assert resp.json()["status"] == "ok"


# ── HTTP error propagation ────────────────────────────────────────────────────


def test_request_raises_on_http_error(client: AppEEARSClient) -> None:
    """``_request`` raises ``AppEEARSError`` on non-2xx, non-403."""
    client._token = "tok"

    mock_resp = _mock_response(400, False, text="Bad request detail")

    with patch.object(client._session, "request", return_value=mock_resp):
        with pytest.raises(AppEEARSError, match="400"):
            client._request("GET", "/task/t-1")
