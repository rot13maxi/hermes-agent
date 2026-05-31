"""Lightweight JSON schema validation — stdlib only.

Validates agent responses against a simplified JSON Schema subset covering:
- type checking (string, number, integer, boolean, array, object, null)
- required fields (for objects)
- enum values
- nested object schemas
- array item schemas
- property schemas

This is intentionally not a full JSON Schema implementation. It covers the
patterns needed to enforce structured output from agent responses.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_response(text: str, schema: Dict[str, Any]) -> Optional[str]:
    """Validate *text* as JSON matching *schema*.

    Returns ``None`` on success, or a human-readable error string on failure.

    The text may be wrapped in markdown code fences (`` ```json ... ``` ``)
    and will be stripped automatically.
    """
    if not text or not isinstance(text, str):
        return "Response is empty."

    stripped = _strip_code_fences(text.strip())

    try:
        parsed = _parse_json(stripped)
    except ValueError as exc:
        return f"Not valid JSON: {exc}"

    errors: List[str] = []
    _validate(parsed, schema, errors, path="$")
    return "; ".join(errors) if errors else None


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _strip_code_fences(s: str) -> str:
    """Remove markdown code fences if present.

    Handles patterns like `````json
    { ... }
    ``````, bare `````{...}```````, and code fences without language hints.
    """
    if not s.startswith("```"):
        return s

    # Find opening fence end (after ``` and optional language tag)
    first_nl = s.find("\n")
    if first_nl != -1:
        # Opening fence is on its own line — content starts after newline
        inner = s[first_nl + 1 :]
    else:
        # Inline fence like ```json {...}```
        inner = s[3:].strip()
        # Strip optional language tag (e.g. "json " or "python ")
        space_idx = inner.find(" ")
        if space_idx != -1 and inner[:space_idx].isalpha():
            inner = inner[space_idx + 1 :]

    # Find closing fence
    end_fence = inner.rfind("```")
    if end_fence != -1:
        inner = inner[:end_fence].rstrip("\n").strip()
    else:
        inner = inner.strip()

    return inner


def _parse_json(s: str) -> Any:
    """Parse JSON with informative errors."""
    import json

    try:
        return json.loads(s)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{exc.msg} (line {exc.lineno}, column {exc.colno})")


def _validate(value: Any, schema: Dict[str, Any], errors: List[str], path: str) -> None:
    """Recursively validate *value* against *schema*, appending errors."""

    # --- type ---
    expected_type = schema.get("type")
    if expected_type is not None and not _check_type(value, expected_type):
        errors.append(
            f"{path}: expected type '{expected_type}', got '{_py_type_name(value)}'"
        )
        return  # don't continue checking children if type is wrong

    # --- enum ---
    enum = schema.get("enum")
    if enum is not None and value not in enum:
        errors.append(f"{path}: value {value!r} not in enum {enum}")

    # --- const ---
    const = schema.get("const")
    if const is not None and value != const:
        errors.append(f"{path}: expected const {const!r}, got {value!r}")

    # --- object properties ---
    if expected_type == "object" and isinstance(value, dict):
        _validate_object(value, schema, errors, path)

    # --- array items ---
    if expected_type == "array" and isinstance(value, list):
        _validate_array(value, schema, errors, path)


def _validate_object(
    value: Dict[str, Any], schema: Dict[str, Any], errors: List[str], path: str
) -> None:
    """Validate required fields and property schemas on an object."""

    # required fields
    required: List[str] = schema.get("required", [])
    if not isinstance(required, list):
        required = []
    present: Set[str] = set(value.keys())
    for field in required:
        if field not in present:
            errors.append(f"{path}: missing required field '{field}'")

    # property schemas
    properties: Dict[str, Any] = schema.get("properties", {})
    if not isinstance(properties, dict):
        properties = {}
    for key, prop_schema in properties.items():
        if key in value and isinstance(prop_schema, dict):
            _validate(value[key], prop_schema, errors, f"{path}.{key}")

    # additionalProperties: false — reject unknown keys
    additional = schema.get("additionalProperties")
    if additional is False and isinstance(properties, dict):
        known = set(properties.keys())
        for key in present:
            if key not in known:
                errors.append(f"{path}: unexpected field '{key}'")


def _validate_array(
    value: List[Any], schema: Dict[str, Any], errors: List[str], path: str
) -> None:
    """Validate array items and constraints."""

    # minItems / maxItems
    min_items = schema.get("minItems")
    if min_items is not None and len(value) < min_items:
        errors.append(f"{path}: array has {len(value)} items, minimum is {min_items}")

    max_items = schema.get("maxItems")
    if max_items is not None and len(value) > max_items:
        errors.append(f"{path}: array has {len(value)} items, maximum is {max_items}")

    # items schema
    items_schema = schema.get("items")
    if isinstance(items_schema, dict):
        for i, item in enumerate(value):
            _validate(item, items_schema, errors, f"{path}[{i}]")

    # prefixItems (ordered item schemas)
    prefix_items = schema.get("prefixItems")
    if isinstance(prefix_items, list):
        for i, item_schema in enumerate(prefix_items):
            if i < len(value) and isinstance(item_schema, dict):
                _validate(value[i], item_schema, errors, f"{path}[{i}]")  # type: ignore[arg-type]


def _check_type(value: Any, expected: str) -> bool:
    """Check if a Python value matches a JSON Schema type string."""
    if expected == "string":
        return isinstance(value, str)
    elif expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    elif expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    elif expected == "boolean":
        return isinstance(value, bool)
    elif expected == "array":
        return isinstance(value, list)
    elif expected == "object":
        return isinstance(value, dict)
    elif expected == "null":
        return value is None
    return True  # unknown type — pass through


def _py_type_name(value: Any) -> str:
    """Map Python values to JSON Schema type names."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__
