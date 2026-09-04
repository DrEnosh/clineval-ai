"""Versioned prompt loading.

Prompts live in `prompts/<name>_<version>.txt`, not in Python string literals,
for one reason: a prompt is an experimental variable. A run that cannot say
which prompt produced it is not reproducible, and a prompt edited in place
silently invalidates every prior result that cited its version.

Each file carries YAML front matter (name, version, role) and a body split
into `SYSTEM:` and `USER:` sections. Every loaded prompt records a content
hash, so an edited file is detectable even if its version string was not
bumped — the failure mode version strings alone do not catch.

Rendering uses `str.format`, so literal braces in a prompt body are written
`{{` / `}}`. All substituted content is passed through the sanitizer first
(see app/prompts/sanitize.py).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from app.core.config import BASE_DIR
from app.prompts.sanitize import safe_content

PROMPT_DIR = BASE_DIR / "prompts"

_FRONT_MATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_SECTION_SPLIT = re.compile(r"^USER:\s*$", re.MULTILINE)


class PromptError(RuntimeError):
    """A prompt file is missing or malformed."""


@dataclass(frozen=True)
class Prompt:
    """One versioned prompt template."""

    name: str
    version: str
    role: str
    system: str
    user_template: str
    content_hash: str
    path: Path

    @property
    def identifier(self) -> str:
        return f"{self.name}_{self.version}"

    def render(self, **values: str) -> tuple[str, str]:
        """Render (system, user), sanitizing every substituted value.

        Sanitizing here rather than at each call site means a new caller
        cannot forget to do it — the boundary is enforced in one place.
        """
        safe_values = {key: safe_content(str(value)) for key, value in values.items()}
        try:
            user = self.user_template.format(**safe_values)
        except KeyError as error:
            raise PromptError(
                f"Prompt '{self.identifier}' requires a value for {error} that was not supplied."
            ) from error
        return self.system, user

    def metadata(self) -> dict[str, str]:
        """What a run records about this prompt."""
        return {
            "prompt_name": self.name,
            "prompt_version": self.version,
            "prompt_hash": self.content_hash,
        }


def _parse(text: str, path: Path) -> Prompt:
    match = _FRONT_MATTER.match(text)
    if match is None:
        raise PromptError(f"Prompt {path} is missing its '---' front matter block.")

    metadata: dict[str, str] = {}
    current_key: str | None = None
    for line in match.group(1).splitlines():
        if not line.strip():
            continue
        if line.startswith((" ", "\t")) and current_key:
            metadata[current_key] += " " + line.strip()
            continue
        key, _, value = line.partition(":")
        current_key = key.strip()
        metadata[current_key] = value.strip()

    for required in ("name", "version"):
        if not metadata.get(required):
            raise PromptError(f"Prompt {path} front matter is missing '{required}'.")

    body = text[match.end() :]
    if not body.lstrip().startswith("SYSTEM:"):
        raise PromptError(f"Prompt {path} must begin with a 'SYSTEM:' section.")

    sections = _SECTION_SPLIT.split(body, maxsplit=1)
    if len(sections) != 2:
        raise PromptError(f"Prompt {path} must contain exactly one 'USER:' section.")

    system = sections[0].lstrip()[len("SYSTEM:") :].strip()
    user_template = sections[1].strip()
    if not system or not user_template:
        raise PromptError(f"Prompt {path} has an empty SYSTEM or USER section.")

    return Prompt(
        name=metadata["name"],
        version=metadata["version"],
        role=metadata.get("role", ""),
        system=system,
        user_template=user_template,
        content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
        path=path,
    )


@lru_cache(maxsize=32)
def load_prompt(name: str, version: str, directory: str | None = None) -> Prompt:
    """Load `prompts/<name>_<version>.txt`.

    Cached, because prompt files do not change during a run. Pass a directory
    to load from elsewhere (tests do this).
    """
    base = Path(directory) if directory else PROMPT_DIR
    path = base / f"{name}_{version}.txt"
    if not path.exists():
        available = ", ".join(sorted(p.stem for p in base.glob("*.txt"))) or "none"
        raise PromptError(
            f"Prompt '{name}_{version}' not found in {base}. Available: {available}."
        )
    return _parse(path.read_text(encoding="utf-8"), path)


def available_prompts(directory: str | None = None) -> list[str]:
    base = Path(directory) if directory else PROMPT_DIR
    if not base.exists():
        return []
    return sorted(path.stem for path in base.glob("*.txt"))


def clear_cache() -> None:
    """Drop the prompt cache. Used by tests that write prompt files."""
    load_prompt.cache_clear()
