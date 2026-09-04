"""Deterministic output cache.

Generation is the expensive, non-reproducible half of an evaluation. Caching
it makes a run repeatable and stops a re-evaluation costing money a second
time.

The cache key covers everything that determines the output:

    case_id + provider + model + prompt name + prompt version + prompt hash
            + temperature + max_tokens + run_index

Two things worth noting about that list. The **prompt hash** is included as
well as the version, so editing a prompt file without bumping its version
still invalidates the cache — the failure mode a version string alone does not
catch. And **run_index** is included, so deliberately repeated generations of
one case do not collapse into a single cached entry, which would make
repeatability testing meaningless.

Entries are JSON files on disk under `data/cache/`, gitignored as runtime
state. Reads never raise: a corrupt or unreadable entry is treated as a miss,
because a cache failure should cost a regeneration, not the run.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from app.core.config import settings as default_settings
from app.generation.schemas import ClinicalSummaryRequest, ClinicalSummaryResponse


class OutputCache:
    """File-backed cache of generated summaries."""

    def __init__(self, directory: Path | str | None = None, enabled: bool = True):
        self.directory = Path(directory) if directory else default_settings.cache_dir
        self.enabled = enabled
        self.hits = 0
        self.misses = 0
        self.writes = 0

    @staticmethod
    def build_key(request: ClinicalSummaryRequest, prompt_hash: str) -> str:
        payload = {**request.configuration(), "case_id": request.case_id, "prompt_hash": prompt_hash}
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:32]

    def _path(self, key: str) -> Path:
        return self.directory / f"{key}.json"

    def get(
        self, request: ClinicalSummaryRequest, prompt_hash: str
    ) -> ClinicalSummaryResponse | None:
        """Return a cached response, or None on any miss or read failure."""
        if not self.enabled:
            return None

        path = self._path(self.build_key(request, prompt_hash))
        if not path.exists():
            self.misses += 1
            return None

        try:
            response = ClinicalSummaryResponse.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except Exception:  # noqa: BLE001 - a bad entry is a miss, never a crash
            self.misses += 1
            return None

        self.hits += 1
        return response.model_copy(update={"from_cache": True})

    def put(
        self,
        request: ClinicalSummaryRequest,
        prompt_hash: str,
        response: ClinicalSummaryResponse,
    ) -> None:
        """Store a response. Failed generations are never cached."""
        if not self.enabled or response.error is not None:
            return

        self.directory.mkdir(parents=True, exist_ok=True)
        path = self._path(self.build_key(request, prompt_hash))
        path.write_text(
            response.model_dump_json(indent=2, exclude={"from_cache"}),
            encoding="utf-8",
        )
        self.writes += 1

    def clear(self) -> int:
        """Delete every cached entry. Returns how many were removed."""
        if not self.directory.exists():
            return 0
        removed = 0
        for path in self.directory.glob("*.json"):
            path.unlink()
            removed += 1
        return removed

    def stats(self) -> dict[str, int | float | None]:
        looked_up = self.hits + self.misses
        return {
            "enabled": self.enabled,
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
            "hit_rate": round(self.hits / looked_up, 4) if looked_up else None,
        }
