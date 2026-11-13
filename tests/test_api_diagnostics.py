from livekit.agents import APIConnectionError, APIStatusError

from api_diagnostics import api_error_details


def test_unwrap_status_without_exposing_payload():
    inner = APIStatusError(
        "secret response",
        status_code=503,
        request_id="req-123",
        body={"secret": "private"},
    )
    outer = APIConnectionError("retry exhausted")
    outer.__cause__ = inner
    details = api_error_details(outer)
    assert details["status_code"] == 503
    assert details["request_id"] == "req-123"
    assert details["cause_types"] == ["APIConnectionError", "APIStatusError"]
    assert "secret" not in str(details)


def test_cycle_is_bounded():
    error = APIConnectionError()
    error.__cause__ = error
    assert api_error_details(error)["cause_types"] == ["APIConnectionError"]


def test_timeout_classified():
    assert api_error_details(TimeoutError())["error_type"] == "TimeoutError"
