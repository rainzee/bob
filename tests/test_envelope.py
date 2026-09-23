from dataclasses import dataclass

from bub.envelope import content_of, field_of


@dataclass
class _Message:
    content: str
    channel: str = "cli"


def test_field_of_supports_mapping_and_object() -> None:
    mapping = {"content": "hello", "count": 3}
    assert field_of(mapping, "content") == "hello"
    assert field_of(mapping, "missing", "fallback") == "fallback"

    obj = _Message(content="world")
    assert field_of(obj, "content") == "world"
    assert field_of(obj, "missing", "fallback") == "fallback"


def test_content_of_stringifies_value() -> None:
    assert content_of({"content": 123}) == "123"
    assert content_of({"other": "x"}) == ""
