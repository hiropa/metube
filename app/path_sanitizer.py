"""One place that turns arbitrary metadata into a safe filesystem component.

Every path component MeTube writes (titles, artists, albums, channel names,
chapter titles, ...) is untrusted: it comes from yt-dlp metadata that a site
controls. This module both *repairs* text that arrived mis-decoded and makes a
single name safe for common filesystems (Windows/SMB, ext4, APFS) without
destroying legitimately non-Latin text.

The public entry point is :func:`sanitize_path_component`. Its defaults come
from the environment through :func:`configure`, which ``main.Config`` calls
once at startup.
"""

from __future__ import annotations

import logging
import os
import re
import unicodedata
from typing import Any

log = logging.getLogger('path_sanitizer')

# POSIX guarantees NAME_MAX >= 255; ext4/xfs/btrfs and NTFS all allow 255
# UTF-8 bytes per component.
DEFAULT_MAX_BYTES = 255
_FALLBACK_NAME = '_'

# Windows-reserved device names. They are reserved with *any* extension, and
# compared case-insensitively.
_RESERVED_NAMES = frozenset(
    ['CON', 'PRN', 'AUX', 'NUL']
    + [f'COM{i}' for i in range(1, 10)]
    + [f'LPT{i}' for i in range(1, 10)]
)

# Characters illegal in Windows/NTFS path components, plus C0 controls and
# DEL, U+FFFD (an unrepairable input byte) and lone UTF-16 surrogates, which
# cannot be encoded to UTF-8 at all. '/' and '\\' are handled separately
# (they imply a path, not a title).
_ILLEGAL_RE = re.compile(r'[<>:"|?*\x00-\x1f\x7f\ufffd\ud800-\udfff]')
# C1 controls (0x80-0x9F). In a correctly decoded UTF-8 name these never
# appear; they are a classic sign of a cp1252/latin-1 mix-up.
_C1_RE = re.compile(r'[\x80-\x9f]')
# Sequences typical of UTF-8 bytes that were decoded as latin-1 or cp1252:
# "Ã¤" for ä, "â€™" for ’, "ðŸ" for an emoji, and a UTF-8 BOM read as cp1252.
_MOJIBAKE_RE = re.compile(r'(?:Ã.|Â.|â€.|ðŸ|ï»¿)')
_SEPARATOR_RE = re.compile(r'[\\/]+')

_settings = {
    'fix_encoding': True,
    'replacement': '_',
    'ascii_only': False,
}


def configure(*, fix_encoding: bool = True, replacement: str = '_',
              ascii_only: bool = False) -> None:
    """Set the process-wide defaults used by :func:`sanitize_path_component`."""
    replacement = replacement if replacement is not None else '_'
    _settings['fix_encoding'] = bool(fix_encoding)
    _settings['replacement'] = replacement
    _settings['ascii_only'] = bool(ascii_only)


def _suspicion_score(value: str) -> int:
    """How much evidence there is that *value* is mis-decoded text."""
    return (
        5 * value.count('\ufffd')
        + 2 * len(_MOJIBAKE_RE.findall(value))
        + 2 * len(_C1_RE.findall(value))
        + sum(1 for ch in value if unicodedata.category(ch) == 'Cc')
    )


def _repair_encoding(value: str) -> str:
    """Undo the common "UTF-8 read as latin-1/cp1252" mistakes.

    A candidate is accepted only when it removes suspicious patterns and
    introduces no replacement characters. Correctly encoded text -- Korean,
    Japanese, Cyrillic, Arabic, an umlaut -- either has no suspicious patterns
    or produces no valid candidate, so it is returned untouched.
    """
    before = _suspicion_score(value)
    if before == 0:
        return value

    candidates: list[str] = []
    try:  # Optional: ftfy knows more heuristics than this module needs.
        import ftfy  # pylint: disable=import-outside-toplevel
    except ImportError:
        ftfy = None
    if ftfy is not None:
        candidates.append(ftfy.fix_text(value))

    for encoding in ('latin-1', 'cp1252'):
        try:
            fixed = value.encode(encoding).decode('utf-8')
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
        candidates.append(fixed)
        # Text that was decoded wrong twice ("ÃƒÂ¤").
        try:
            candidates.append(fixed.encode(encoding).decode('utf-8'))
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue

    if not candidates:
        return value
    best = min(candidates, key=_suspicion_score)
    if _suspicion_score(best) < before and '\ufffd' not in best:
        return best
    return value


def _truncate_bytes(value: str, max_bytes: int) -> str:
    """Shorten *value* to *max_bytes* of UTF-8, keeping a short extension."""
    if len(value.encode('utf-8', 'replace')) <= max_bytes:
        return value
    stem, ext = os.path.splitext(value)
    ext_bytes = ext.encode('utf-8', 'replace')
    if len(ext_bytes) > 16:  # not really an extension
        stem, ext, ext_bytes = value, '', b''
    budget = max(max_bytes - len(ext_bytes), 1)
    # A character boundary may not survive the cut; 'ignore' drops the partial
    # character rather than emitting a replacement character.
    trimmed = stem.encode('utf-8', 'replace')[:budget].decode('utf-8', 'ignore').rstrip()
    return (trimmed or _FALLBACK_NAME) + ext


def sanitize_path_component(
    name: str,
    *,
    replacement: str | None = None,
    ascii_only: bool | None = None,
    fix_encoding: bool | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> str:
    """Return *name* as one safe file/directory component.

    Non-string values are returned unchanged so yt-dlp numeric format specs
    keep working (the callers only pass strings here, but the guard keeps that
    contract explicit). Path separators never survive: a value that contains
    them is treated as a path and its meaningful segments are joined with the
    replacement character, dropping '.' and '..' segments. Interior dots in a
    plain title ("Wait... what?") are preserved.
    """
    if not isinstance(name, str):
        return name

    replacement = _settings['replacement'] if replacement is None else replacement
    ascii_only = _settings['ascii_only'] if ascii_only is None else ascii_only
    fix_encoding = _settings['fix_encoding'] if fix_encoding is None else fix_encoding

    original = name
    name = unicodedata.normalize('NFC', name)

    if fix_encoding:
        name = unicodedata.normalize('NFC', _repair_encoding(name))

    if ascii_only:
        # NFKD splits accents into a base letter plus a combining mark, so
        # "Beyoncé" becomes "Beyonce"; scripts without an ASCII form are
        # removed by the ASCII encode.
        name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('ascii')

    if any(sep in name for sep in ('/', '\\')):
        leading_separator = name[0] in '/\\'
        segments = [
            segment for segment in _SEPARATOR_RE.split(name)
            if segment not in ('', '.', '..')
        ]
        joined = replacement.join(segments)
        name = (replacement if leading_separator else '') + joined

    name = _ILLEGAL_RE.sub(replacement, name)
    name = _C1_RE.sub(replacement, name)

    # Trailing dots and spaces are stripped by Windows and break SMB/NAS
    # round trips. A component of only dots ('.', '..', '...') is traversal
    # debris and becomes the fallback; a leading dot on a real name is kept,
    # because it may be an operator-chosen folder such as '.metube-videos'.
    name = name.strip().rstrip('.')

    if not name or set(name) == {'.'}:
        name = _FALLBACK_NAME

    stem = name.split('.', 1)[0].upper()
    if stem in _RESERVED_NAMES:
        prefix = replacement or _FALLBACK_NAME
        name = prefix + name

    result = _truncate_bytes(name, max_bytes)
    if result != original:
        log.warning('Sanitized path component %r -> %r', original, result)
    return result


def sanitize_component_safe(value: Any) -> Any:
    """``sanitize_path_component`` for values of unknown type (non-strings pass through)."""
    return sanitize_path_component(value) if isinstance(value, str) else value
