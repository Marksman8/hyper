"""Argument validation against a tool's JSON Schema.

A deliberate subset — `type`, `required`, `properties`, `enum`, `minimum`,
`maximum`, `minLength`, `maxLength`, `items` — rather than the `jsonschema`
package, because those are the only keywords AURA's tool schemas use and spec
section 30 says not to add dependencies without a reason.

**This is not the security boundary.** It exists to give a model a useful error
when it passes a string where an integer was wanted, and to stop type confusion
reaching tool code. What actually protects the filesystem is
`workspace.Workspace`, and what protects against dangerous operations is the
permission level. If this validator let something through, those still hold.

If tool schemas ever need `oneOf`, `$ref`, or conditional subschemas, replace
this with `jsonschema` rather than growing it — a half-implemented validator is
worse than an honest dependency.
"""

from __future__ import annotations

from typing import Any

# JSON Schema type name -> Python types. `bool` is checked before `int`
# everywhere below, because in Python `True` is an instance of `int` and a
# schema asking for a number should not silently accept a boolean.
_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
    "null": (type(None),),
}


def _type_error(name: str, expected: str, value: Any) -> str:
    return f"{name!r} must be {expected}, got {type(value).__name__}"


def _check_type(name: str, expected: str, value: Any) -> str | None:
    allowed = _TYPES.get(expected)
    if allowed is None:
        return None  # unknown keyword: not our job to guess
    if expected in ("integer", "number") and isinstance(value, bool):
        return _type_error(name, expected, value)
    if not isinstance(value, allowed):
        return _type_error(name, expected, value)
    return None


def _check_one(name: str, schema: dict[str, Any], value: Any) -> str | None:
    expected = schema.get("type")
    if isinstance(expected, str):
        err = _check_type(name, expected, value)
        if err:
            return err
    elif isinstance(expected, list):
        if all(_check_type(name, t, value) is not None for t in expected):
            return _type_error(name, " or ".join(expected), value)

    if "enum" in schema and value not in schema["enum"]:
        return f"{name!r} must be one of {schema['enum']}, got {value!r}"

    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            return f"{name!r} must be at least {schema['minLength']} characters"
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            return f"{name!r} must be at most {schema['maxLength']} characters"

    if isinstance(value, int | float) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            return f"{name!r} must be >= {schema['minimum']}, got {value}"
        if "maximum" in schema and value > schema["maximum"]:
            return f"{name!r} must be <= {schema['maximum']}, got {value}"

    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        for i, item in enumerate(value):
            err = _check_one(f"{name}[{i}]", schema["items"], item)
            if err:
                return err

    return None


def validate_arguments(schema: dict[str, Any], arguments: dict[str, Any]) -> str | None:
    """Return the first problem with `arguments`, or None if they are fine.

    One message rather than a list: it goes to a model as an observation, and
    a single specific correction is acted on more reliably than a report.
    """
    if not isinstance(arguments, dict):
        return f"arguments must be an object, got {type(arguments).__name__}"

    properties: dict[str, Any] = schema.get("properties", {})

    # Unknown arguments are refused rather than ignored, and are checked before
    # missing ones because an unknown argument usually explains a missing one:
    # a model that sent `paht` is better told about the typo than told that
    # `path` is absent.
    if schema.get("additionalProperties") is False:
        for name in arguments:
            if name not in properties:
                return f"unknown argument {name!r}; expected one of {sorted(properties) or 'none'}"

    for name in schema.get("required", []):
        if name not in arguments:
            return f"missing required argument {name!r}"

    for name, value in arguments.items():
        sub = properties.get(name)
        if isinstance(sub, dict):
            err = _check_one(name, sub, value)
            if err:
                return err

    return None
