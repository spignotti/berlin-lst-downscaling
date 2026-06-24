"""Unit tests for the AppEEARS REST API client.

All tests mock the HTTP layer — no real API calls.
"""

from __future__ import annotations

from pathlib import Path
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


# ── Auth ──────────────────────────────────────────────────────────────────────


def test_login_success(client: AppEEARSClient) -> None:
    """Login returns a token on valid credentials."""
    mock_resp = _mock_response(200, True, {"token": "abc123"})

    with patch.object(client._session, "post", return_value=mock_resp) as mock_post:
        token = client.login("user", "pass")

    assert token == "abc123"
    assert client._token == "abc123"
    mock_post.assert_called_once()


def test_login_invalid_credentials(client: AppEEARSClient) -> None:
    """Login raises AuthError on 401."""
    mock_resp = _mock_response(401)

    with patch.object(client._session, "post", return_value=mock_resp):
        with pytest.raises(AuthError, match="Invalid Earthdata credentials"):
            client.login("user", "wrong")


def test_login_http_error(client: AppEEARSClient) -> None:
    """Login raises on non-401 HTTP error."""
    mock_resp = _mock_response(500, False)

    with patch.object(client._session, "post", return_value=mock_resp):
        with pytest.raises(AppEEARSError):
            client.login("user", "pass")


def test_logout_clears_token(client: AppEEARSClient) -> None:
    """Logout clears the cached token."""
    client._token = "abc123"
    mock_resp = _mock_response(204, True)

    with patch.object(client._session, "post", return_value=mock_resp):
        client.logout()

    assert client._token is None


def test_logout_noop_when_not_logged_in(client: AppEEARSClient) -> None:
    """Logout does nothing when no token is cached."""
    client.logout()  # Should not raise


# ── Auth header ───────────────────────────────────────────────────────────────


def test_auth_header_raises_when_not_logged_in(client: AppEEARSClient) -> None:
    """_auth_header raises AuthError if no token."""
    with pytest.raises(AuthError, match="Not logged in"):
        client._auth_header()


def test_auth_header_returns_bearer_token(client: AppEEARSClient) -> None:
    """_auth_header returns correct Authorization header."""
    client._token = "abc123"
    headers = client._auth_header()
    assert headers == {"Authorization": "Bearer abc123"}


# ── _ensure_auth from env ─────────────────────────────────────────────────────


def test_ensure_auth_from_env(client: AppEEARSClient) -> None:
    """_ensure_auth reads credentials from env and logs in."""
    mock_resp = _mock_response(200, True, {"token": "env_token"})

    with patch.object(client._session, "post", return_value=mock_resp):
        env = {"EARTHDATA_USERNAME": "envuser", "EARTHDATA_PASSWORD": "envpass"}
        with patch.dict("os.environ", env):
            client._ensure_auth()
            assert client._token == "env_token"


def test_ensure_auth_missing_env(client: AppEEARSClient) -> None:
    """_ensure_auth raises if env vars are missing."""
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(AuthError, match="EARTHDATA_USERNAME"):
            client._ensure_auth()


# ── Task submission ───────────────────────────────────────────────────────────


def test_submit_area_task(client: AppEEARSClient) -> None:
    """Submit area task returns the task ID."""
    client._token = "tok"

    mock_resp = _mock_response(202, True, {"task_id": "task-123", "status": "pending"})

    geo_json = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
                },
                "properties": {},
            }
        ],
    }
    layers = [{"product": "ECO_L2T_LSTE.002", "layer": "LST"}]
    dates = [
        {"startDate": "05-01", "endDate": "09-30", "recurring": True, "yearRange": [2023, 2023]}
    ]

    with patch.object(client._session, "request", return_value=mock_resp) as mock_req:
        task_id = client.submit_area_task("test-task", geo_json, layers, dates)

    assert task_id == "task-123"
    call_kwargs = mock_req.call_args[1]
    assert call_kwargs["json"]["task_type"] == "area"
    assert call_kwargs["json"]["params"]["geo"] == geo_json


def test_submit_area_task_no_task_id(client: AppEEARSClient) -> None:
    """Raises TaskError if response has no task_id."""
    client._token = "tok"

    mock_resp = _mock_response(200, True, {"status": "error"})

    with patch.object(client._session, "request", return_value=mock_resp):
        with pytest.raises(TaskError, match="did not return a task_id"):
            client.submit_area_task("test", {}, [], [])


# ── Task status ───────────────────────────────────────────────────────────────


def test_get_task_status_returns_dict(client: AppEEARSClient) -> None:
    """get_task_status returns the parsed JSON."""
    client._token = "tok"

    mock_resp = _mock_response(200, True, {"status": "done", "task_id": "t-1"})

    with patch.object(client._session, "request", return_value=mock_resp):
        status = client.get_task_status("t-1")

    assert status["status"] == "done"


# ── Wait for task ─────────────────────────────────────────────────────────────


def test_wait_for_task_done(client: AppEEARSClient) -> None:
    """wait_for_task returns when status is 'done'."""
    client._token = "tok"

    mock_resp = _mock_response(200, True, {"status": "done", "task_id": "t-1"})

    with patch.object(client._session, "request", return_value=mock_resp):
        result = client.wait_for_task("t-1", poll_interval_sec=1, timeout_hours=1)

    assert result["status"] == "done"


def test_wait_for_task_error(client: AppEEARSClient) -> None:
    """wait_for_task raises on task error."""
    client._token = "tok"

    mock_resp = _mock_response(200, True, {"status": "error", "error": "download failed"})

    with patch.object(client._session, "request", return_value=mock_resp):
        with pytest.raises(TaskError, match="download failed"):
            client.wait_for_task("t-1", poll_interval_sec=1, timeout_hours=1)


# ── Bundle / download ─────────────────────────────────────────────────────────


def test_list_bundle_files(client: AppEEARSClient) -> None:
    """list_bundle_files returns the files list."""
    client._token = "tok"

    mock_resp = _mock_response(
        200, True, {"files": [{"file_id": "f1", "file_name": "test.tif", "sha256": "abc123"}]}
    )

    with patch.object(client._session, "request", return_value=mock_resp):
        files = client.list_bundle_files("t-1")

    assert len(files) == 1
    assert files[0]["file_id"] == "f1"


def test_download_file(client: AppEEARSClient, tmp_path: Path) -> None:
    """download_file saves the file to disk."""
    client._token = "tok"

    # Bundle listing response
    bundle_resp = _mock_response(
        200, True, {"files": [{"file_id": "f1", "file_name": "test.tif", "sha256": None}]}
    )

    # Stream download response (simulates requests.get stream)
    stream_resp = MagicMock()
    stream_resp.__enter__.return_value = stream_resp
    stream_resp.ok = True
    stream_resp.raise_for_status.return_value = None
    stream_resp.iter_content.return_value = [b"test-data"]

    with patch.object(client._session, "request", return_value=bundle_resp):
        with patch.object(client._session, "get", return_value=stream_resp):
            dest = client.download_file("t-1", "f1", tmp_path / "out.tif", verify_sha256=False)

    assert dest.exists()
    assert dest.read_bytes() == b"test-data"


# ── Auto-auth 403 retry ───────────────────────────────────────────────────────


def test_auto_reauth_on_403(client: AppEEARSClient) -> None:
    """_request re-authenticates on 403 and retries."""
    client._token = "expired"

    # First call returns 403, second succeeds
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


# ── HTTP error handling ───────────────────────────────────────────────────────


def test_request_raises_on_http_error(client: AppEEARSClient) -> None:
    """_request raises AppEEARSError on non-2xx, non-403."""
    client._token = "tok"

    mock_resp = _mock_response(400, False, text="Bad request detail")

    with patch.object(client._session, "request", return_value=mock_resp):
        with pytest.raises(AppEEARSError, match="400"):
            client._request("GET", "/task/t-1")
