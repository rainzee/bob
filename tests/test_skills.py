from pathlib import Path

import pytest

from bub.builtin.context import default_tape_context
from bub.skills import (
    SKILL_FILE_NAME,
    SkillMetadata,
    _parse_frontmatter,
    _read_skill,
    discover_skills,
    render_skills_prompt,
)


def _write_skill(
    root: Path,
    name: str,
    *,
    description: str = "A skill",
    body: str = "Skill body",
    metadata: dict[str, str] | None = None,
) -> Path:
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "---",
        f"name: {name}",
        f"description: {description}",
    ]
    if metadata is not None:
        lines.append("metadata:")
        for key, value in metadata.items():
            lines.append(f"  {key}: {value}")
    lines.extend(["---", body])
    skill_file = skill_dir / SKILL_FILE_NAME
    skill_file.write_text("\n".join(lines), encoding="utf-8")
    return skill_file


def test_skill_metadata_body_strips_frontmatter(tmp_path: Path) -> None:
    skill_file = _write_skill(tmp_path, "demo-skill", body="Line 1\nLine 2")
    metadata = SkillMetadata(name="demo-skill", description="Demo", location=skill_file)

    assert metadata.body() == "Line 1\nLine 2"


def test_skill_metadata_body_expands_string_templates(tmp_path: Path) -> None:
    skill_file = _write_skill(tmp_path, "demo-skill", body="Skill dir: $SKILL_DIR\nPython: $PYTHON")
    metadata = SkillMetadata(name="demo-skill", description="Demo", location=skill_file)

    body = metadata.body()

    assert f"Skill dir: {tmp_path / 'demo-skill'}" in body
    assert "Python: " in body


def test_skill_metadata_body_leaves_unknown_placeholders_alone(tmp_path: Path) -> None:
    skill_file = _write_skill(tmp_path, "demo-skill", body='Token: "$missing" and ${not-a-template}')
    metadata = SkillMetadata(name="demo-skill", description="Demo", location=skill_file)

    assert metadata.body() == 'Token: "$missing" and ${not-a-template}'


def test_read_skill_rejects_invalid_metadata_field_type(tmp_path: Path) -> None:
    skill_dir = tmp_path / "bad-skill"
    skill_dir.mkdir()
    content = "---\nname: bad-skill\ndescription: bad\nmetadata:\n  retries: 3\n---\nBody\n"
    (skill_dir / SKILL_FILE_NAME).write_text(content, encoding="utf-8")

    assert _read_skill(skill_dir) is None


def test_read_skill_accepts_string_metadata(tmp_path: Path) -> None:
    _write_skill(tmp_path, "with-metadata", metadata={"owner": "team"})

    metadata = _read_skill(tmp_path / "with-metadata")

    assert metadata is not None
    assert metadata.metadata["metadata"] == {"owner": "team"}


def test_discover_skills_scans_the_given_roots_only(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_skill(first, "shared", description="first version")
    _write_skill(first, "first-only")
    _write_skill(second, "shared", description="second version")
    _write_skill(second, "second-only")

    discovered = {skill.name: skill.description for skill in discover_skills([first, second])}

    assert discovered == {"shared": "first version", "first-only": "A skill", "second-only": "A skill"}


def test_discover_skills_ignores_missing_roots(tmp_path: Path) -> None:
    assert discover_skills([tmp_path / "absent"]) == []


def test_discover_skills_without_roots_finds_nothing(tmp_path: Path) -> None:
    _write_skill(tmp_path, "ignored")

    assert discover_skills([]) == []


def test_render_skills_prompt_includes_expanded_body(tmp_path: Path) -> None:
    skill_file = _write_skill(tmp_path, "skill-a", body="Body A")
    skills = [
        SkillMetadata(name="skill-a", description="A", location=skill_file),
        SkillMetadata(name="skill-b", description="B", location=skill_file),
    ]

    rendered = render_skills_prompt(skills, expanded_skills={"skill-a"})

    assert "<available_skills>" in rendered
    assert "- skill-a: A" in rendered
    assert "Location:" in rendered
    assert "Body A" in rendered
    assert "- skill-b: B" in rendered
    assert "Body A" not in rendered.split("- skill-b: B")[1]


def test_render_skills_prompt_is_empty_without_skills() -> None:
    assert render_skills_prompt([]) == ""


def test_parse_frontmatter_requires_leading_delimiter() -> None:
    assert _parse_frontmatter("no frontmatter") == {}


def test_parse_frontmatter_returns_lowercased_keys() -> None:
    assert _parse_frontmatter("---\nName: x\n---\nbody") == {"name": "x"}


def test_default_tape_context_selects_messages() -> None:
    context = default_tape_context()

    assert context.select is not None


@pytest.mark.parametrize("name", ["UPPER", "has space", "trailing-", "-leading"])
def test_read_skill_rejects_invalid_names(tmp_path: Path, name: str) -> None:
    skill_dir = tmp_path / "valid-dir"
    skill_dir.mkdir()
    (skill_dir / SKILL_FILE_NAME).write_text(
        f"---\nname: {name}\ndescription: ok\n---\nBody\n",
        encoding="utf-8",
    )

    assert _read_skill(skill_dir) is None
