import json
from typing import Any


def serialize_tool_result(result: Any) -> str:
    if isinstance(result, str):
        return result

    if result is None:
        return "null"

    try:
        return json.dumps(result, ensure_ascii=False)
    except (TypeError, OverflowError):
        return str(result)
