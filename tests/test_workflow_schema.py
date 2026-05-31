"""Tests for hermes_workflow schema validation."""

import pytest
from hermes_workflow.schema import validate_response, _strip_code_fences


class TestStripCodeFences:
    def test_no_fences(self):
        assert _strip_code_fences('{"a": 1}') == '{"a": 1}'

    def test_json_fenced(self):
        raw = '```json\n{"a": 1}\n```'
        assert _strip_code_fences(raw) == '{"a": 1}'

    def test_bare_fenced(self):
        raw = '```\n{"a": 1}\n```'
        assert _strip_code_fences(raw) == '{"a": 1}'

    def test_inline_fence(self):
        raw = '```json {"a": 1}```'
        assert _strip_code_fences(raw) == '{"a": 1}'


class TestValidateResponse:
    def test_empty(self):
        assert validate_response("", {"type": "string"}) is not None

    def test_none(self):
        assert validate_response(None, {"type": "string"}) is not None

    def test_valid_string(self):
        assert validate_response('"hello"', {"type": "string"}) is None

    def test_valid_integer(self):
        assert validate_response("42", {"type": "integer"}) is None

    def test_valid_boolean(self):
        assert validate_response("true", {"type": "boolean"}) is None

    def test_valid_null(self):
        assert validate_response("null", {"type": "null"}) is None

    def test_type_mismatch(self):
        err = validate_response("42", {"type": "string"})
        assert err is not None
        assert "string" in err

    def test_valid_object(self):
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name"],
        }
        assert validate_response('{"name": "Alice"}', schema) is None
        assert validate_response('{"name": "Bill", "age": 30}', schema) is None
        assert validate_response('{"name": "Claire"}', schema) is None

    def test_missing_required(self):
        schema = {
            "type": "object",
            "required": ["name", "email"],
        }
        err = validate_response('{"name": "Alice"}', schema)
        assert err is not None
        assert "email" in err
        err = validate_response('{"name": "Bill"}', schema)
        assert err is not None
        assert "email" in err

    def test_valid_array(self):
        schema = {
            "type": "array",
            "items": {"type": "string"},
        }
        assert validate_response('["a", "b", "c"]', schema) is None

    def test_array_type_mismatch(self):
        schema = {
            "type": "array",
            "items": {"type": "string"},
        }
        err = validate_response('["a", 42, "c"]', schema)
        assert err is not None
        assert "[1]" in err

    def test_enum_valid(self):
        schema = {
            "type": "string",
            "enum": ["low", "medium", "high"],
        }
        assert validate_response('"high"', schema) is None

    def test_enum_invalid(self):
        schema = {
            "type": "string",
            "enum": ["low", "medium", "high"],
        }
        err = validate_response('"critical"', schema)
        assert err is not None

    def test_const_valid(self):
        assert (
            validate_response('"exact"', {"type": "string", "const": "exact"}) is None
        )

    def test_const_invalid(self):
        err = validate_response('"wrong"', {"type": "string", "const": "exact"})
        assert err is not None

    def test_min_items(self):
        schema = {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 3,
        }
        err = validate_response('["a", "b"]', schema)
        assert err is not None
        assert "minimum" in err

    def test_max_items(self):
        schema = {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 2,
        }
        err = validate_response('["a", "b", "c"]', schema)
        assert err is not None
        assert "maximum" in err

    def test_nested_object(self):
        schema = {
            "type": "object",
            "properties": {
                "file": {"type": "string"},
                "endpoints": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "has_auth": {"type": "boolean"},
                        },
                        "required": ["path", "has_auth"],
                    },
                },
                "severity": {
                    "type": "string",
                    "enum": ["low", "medium", "high", "critical"],
                },
            },
            "required": ["file", "endpoints", "severity"],
        }
        valid = (
            '{"file": "routes/auth.ts", '
            '"endpoints": [{"path": "/login", "has_auth": false}], '
            '"severity": "high"}'
        )
        assert validate_response(valid, schema) is None

    def test_nested_missing_required(self):
        schema = {
            "type": "object",
            "properties": {
                "endpoints": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "has_auth": {"type": "boolean"},
                        },
                        "required": ["path", "has_auth"],
                    },
                },
            },
        }
        err = validate_response('{"endpoints": [{"path": "/x"}]}', schema)
        assert err is not None
        assert "has_auth" in err

    def test_additional_properties_false(self):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "additionalProperties": False,
        }
        err = validate_response('{"name": "Alice", "extra": 1}', schema)
        assert err is not None
        assert "extra" in err

    def test_prefix_items(self):
        schema = {
            "type": "array",
            "prefixItems": [
                {"type": "string"},
                {"type": "integer"},
            ],
        }
        assert validate_response('["hello", 42]', schema) is None
        err = validate_response('[42, "hello"]', schema)
        assert err is not None

    def test_code_fence_stripped(self):
        schema = {"type": "string"}
        assert validate_response('```json\n"hello"\n```', schema) is None

    def test_invalid_json(self):
        err = validate_response("{bad json}", {"type": "object"})
        assert err is not None
        assert "JSON" in err

    def test_boolean_not_integer(self):
        err = validate_response("true", {"type": "integer"})
        assert err is not None

    def test_number_accepts_int_and_float(self):
        assert validate_response("42", {"type": "number"}) is None
        assert validate_response("3.14", {"type": "number"}) is None

    def test_number_rejects_boolean(self):
        err = validate_response("true", {"type": "number"})
        assert err is not None
