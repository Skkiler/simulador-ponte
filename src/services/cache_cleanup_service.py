from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Dict, List


class CacheCleanupService:
    """Limpa caches residuais do projeto sem tocar no ambiente virtual."""

    def __init__(self, project_root: str | Path) -> None:
        self.root = Path(project_root).resolve()
        self.skip_dir_names = {
            ".git",
            "venv",
            ".venv",
            "env",
            "ENV",
            "node_modules",
        }
        self.extra_cache_dirs = [
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
            ".hypothesis",
            ".tox",
            ".nox",
            ".streamlit/cache",
        ]

    def cleanup_filesystem_cache(self) -> Dict[str, List[str] | int]:
        removed_dirs: List[str] = []
        removed_files: List[str] = []

        for current, dirs, files in os.walk(self.root, topdown=True):
            dirs[:] = [d for d in dirs if d not in self.skip_dir_names]

            if "__pycache__" in dirs:
                cache_dir = Path(current) / "__pycache__"
                try:
                    shutil.rmtree(cache_dir, ignore_errors=True)
                    removed_dirs.append(str(cache_dir.relative_to(self.root)))
                except Exception:
                    pass
                dirs.remove("__pycache__")

            for file_name in files:
                if not file_name.endswith((".pyc", ".pyo")):
                    continue

                fpath = Path(current) / file_name
                try:
                    fpath.unlink(missing_ok=True)
                    removed_files.append(str(fpath.relative_to(self.root)))
                except Exception:
                    pass

        for rel in self.extra_cache_dirs:
            cdir = self.root / rel
            if not cdir.exists():
                continue
            try:
                shutil.rmtree(cdir, ignore_errors=True)
                removed_dirs.append(str(cdir.relative_to(self.root)))
            except Exception:
                pass

        return {
            "removed_dirs_count": len(removed_dirs),
            "removed_files_count": len(removed_files),
            "removed_dirs": removed_dirs,
            "removed_files": removed_files,
        }
