"""
Zemo application-level security utilities.

Covers path-traversal prevention, filename sanitisation,
and safe storage-path generation.
"""

from __future__ import annotations

import os
import re
import uuid
from pathlib import Path

from app.core.exceptions import PathTraversalError

# Characters allowed in sanitised filenames
_SAFE_FILENAME_RE = re.compile(r"[^a-zA-Z0-9._-]")

# Extensions that may carry executable payloads — always rejected
BLOCKED_EXTENSIONS: frozenset[str] = frozenset({
    ".exe", ".bat", ".cmd", ".com", ".msi", ".ps1", ".sh",
    ".py", ".rb", ".pl", ".php", ".jar", ".class",
    ".dll", ".so", ".dylib",
    ".pkl", ".pickle",  # Unsafe deserialisation
})

ALLOWED_DATASET_EXTENSIONS: frozenset[str] = frozenset({".csv", ".json", ".jsonl"})


def sanitize_filename(name: str) -> str:
    """
    Return a filesystem-safe version of *name*.

    Strips directory components, replaces unsafe characters,
    and rejects blocked extensions.
    """
    # Remove any directory component
    name = os.path.basename(name)
    # Replace sequences of unsafe chars with a single underscore
    name = _SAFE_FILENAME_RE.sub("_", name).strip("_")
    if not name:
        name = "unnamed"
    return name


def generate_storage_id() -> str:
    """Generate a unique, opaque storage identifier."""
    return uuid.uuid4().hex


def safe_join(base: Path, *parts: str) -> Path:
    """
    Join *parts* onto *base* and verify the result stays within *base*.

    Raises PathTraversalError if the resolved path escapes the base.
    """
    joined = base.joinpath(*parts).resolve()
    base_resolved = base.resolve()
    if not str(joined).startswith(str(base_resolved)):
        raise PathTraversalError(
            "Path traversal attempt detected.",
            detail={"base": str(base_resolved)},
        )
    return joined


def validate_extension(filename: str) -> str:
    """
    Return the lowercased extension if it is an allowed dataset format.

    Raises UnsupportedFormatError otherwise.
    """
    from app.core.exceptions import UnsupportedFormatError

    _, ext = os.path.splitext(filename)
    ext = ext.lower()
    if ext in BLOCKED_EXTENSIONS:
        raise UnsupportedFormatError(
            f"File extension '{ext}' is blocked for security reasons.",
            detail={"extension": ext},
        )
    if ext not in ALLOWED_DATASET_EXTENSIONS:
        raise UnsupportedFormatError(
            f"Unsupported dataset format '{ext}'. Supported: {', '.join(sorted(ALLOWED_DATASET_EXTENSIONS))}",
            detail={"extension": ext, "supported": sorted(ALLOWED_DATASET_EXTENSIONS)},
        )
    return ext
