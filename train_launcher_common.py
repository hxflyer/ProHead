from __future__ import annotations

import sys
from typing import Iterable


def current_platform_key() -> str:
    return "windows" if sys.platform.startswith("win") else "linux"


def resolve_platform_key(platform_key: str | None = None) -> str:
    return str(platform_key or current_platform_key()).lower()


def launch_for_platform(args, launch_linux, launch_windows, platform_key: str | None = None):
    resolved = resolve_platform_key(platform_key)
    if resolved == "windows":
        return launch_windows(args)
    return launch_linux(args)


def resolve_platform_value(values_by_platform: dict[str, object], platform_key: str | None = None):
    resolved = resolve_platform_key(platform_key)
    if resolved in values_by_platform:
        return values_by_platform[resolved]
    if "default" in values_by_platform:
        return values_by_platform["default"]
    available = ", ".join(sorted(values_by_platform.keys()))
    raise KeyError(f"No preset configured for platform={resolved!r}. Available: {available}")


def data_roots_from_folders(data_folders: Iterable[object]) -> list[str]:
    roots: list[str] = []
    for item in data_folders:
        if isinstance(item, tuple):
            roots.append(str(item[0]))
        else:
            roots.append(str(item))
    return roots


def synthetic_data_roots_from_folders(data_folders: Iterable[object]) -> list[str]:
    roots: list[str] = []
    for item in data_folders:
        if isinstance(item, tuple) and len(item) >= 2 and str(item[1]) == "synthetic":
            roots.append(str(item[0]))
    return roots
