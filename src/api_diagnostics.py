"""Safe provider diagnostics: preserve causes without logging prompts or bodies."""


def api_error_details(error: BaseException) -> dict:
    details = {"error_type": type(error).__name__, "cause_types": []}
    seen = set()
    current = error
    while current is not None and id(current) not in seen and len(seen) < 10:
        seen.add(id(current))
        details["cause_types"].append(type(current).__name__)
        for key in ("status_code", "request_id", "retryable"):
            value = getattr(current, key, None)
            if value is not None:
                details[key] = value
        current = current.__cause__ or current.__context__
    return details
