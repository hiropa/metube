"""Square cover art for audio downloads.

yt-dlp already downloads a thumbnail, converts it to JPEG and embeds it
(``FFmpegThumbnailsConvertor`` + ``EmbedThumbnail``). This module adds the two
things it does not do: crop the image to 1:1 without stretching, and keep a
copy as ``cover.jpg``/``cover.png`` in the download's folder.

The postprocessor is added to the ``post_process`` stage *before*
``EmbedThumbnail`` (MeTube's download pipeline inserts it there), so the file
yt-dlp embeds is the cropped one. Everything here is supplemental: a failure
logs a warning and never fails a download that otherwise succeeded.
"""

from __future__ import annotations

import io
import logging
import os
import tempfile
from typing import Any, Optional

from PIL import Image
from yt_dlp.postprocessor.common import PostProcessor

import path_sanitizer

log = logging.getLogger('covers')

COVER_QUALITY = 95
_MAX_EXT_BYTES = 16


def normalize_image(image: Image.Image) -> Image.Image:
    """Flatten transparency onto white and return RGB."""
    if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
        image = image.convert('RGBA')
        background = Image.new('RGB', image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[-1])
        return background
    if image.mode != 'RGB':
        return image.convert('RGB')
    return image


def center_square(image: Image.Image) -> Image.Image:
    """Center-crop *image* to the largest possible 1:1 square.

    Never stretches and never upscales: the side is ``min(width, height)``.
    """
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return image.crop((left, top, left + side, top + side))


def render_cover(data: bytes, *, image_format: str = 'jpg',
                 square: bool = True) -> tuple[bytes, int, int]:
    """Decode *data*, optionally center-crop it, and encode it.

    Returns ``(encoded_bytes, width, height)``. The output is always a real
    JPEG or PNG regardless of the input's declared type (thumbnails are often
    WebP served under a wrong extension).
    """
    image_format = (image_format or 'jpg').lower().lstrip('.')
    if image_format == 'jpeg':
        image_format = 'jpg'
    if image_format not in ('jpg', 'png'):
        image_format = 'jpg'

    with Image.open(io.BytesIO(data)) as opened:
        opened.load()
        image = normalize_image(opened)
        if square:
            image = center_square(image)
        output = io.BytesIO()
        if image_format == 'png':
            image.save(output, 'PNG', optimize=True)
        else:
            image.save(output, 'JPEG', quality=COVER_QUALITY)
        return output.getvalue(), image.width, image.height


def _dimension(thumbnail: dict[str, Any], key: str) -> Optional[int]:
    value = thumbnail.get(key)
    if isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _area(thumbnail: dict[str, Any]) -> int:
    width = _dimension(thumbnail, 'width')
    height = _dimension(thumbnail, 'height')
    return width * height if width and height else 0


def select_cover_source(thumbnails: Any) -> Optional[dict[str, Any]]:
    """Pick the highest-resolution square thumbnail, else the best one.

    yt-dlp puts the preferred thumbnail last, so ties resolve to the later
    entry. Entries with neither a URL nor a local file are ignored.
    """
    if not isinstance(thumbnails, list):
        return None
    usable = [
        t for t in thumbnails
        if isinstance(t, dict) and (t.get('url') or t.get('filepath'))
    ]
    if not usable:
        return None
    squares = [
        t for t in usable
        if _dimension(t, 'width') and _dimension(t, 'width') == _dimension(t, 'height')
    ]
    pool = squares or usable
    best_index = max(range(len(pool)), key=lambda index: (_area(pool[index]), index))
    return pool[best_index]


def _sanitize_cover_stem(value: str) -> str:
    """Keep the configured cover name inside one path component."""
    stem = os.path.basename((value or 'cover').replace('\\', '/'))
    stem = os.path.splitext(stem)[0] or 'cover'
    # Same rules as any other filename: no separators/traversal, no reserved
    # Windows names, encoding repaired. 200 bytes leaves room for '.jpg'.
    return path_sanitizer.sanitize_path_component(stem, max_bytes=200)


def atomic_write(directory: str, filename: str, data: bytes) -> str:
    """Write *data* to ``directory/filename`` without exposing a partial file."""
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.metube-cover-', dir=directory)
    try:
        with os.fdopen(descriptor, 'wb') as handle:
            handle.write(data)
        destination = os.path.join(directory, filename)
        os.replace(temporary, destination)
        return destination
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


class SquareCoverArtPP(PostProcessor):
    """Crop the chosen thumbnail to 1:1 and optionally save a cover file.

    Runs in the ``post_process`` stage before ``EmbedThumbnail`` so the
    embedded image is the cropped one. The cover file is written into
    ``info['__finaldir']`` -- the directory the media will end up in -- so it
    lands next to the audio file even when ``TEMP_DIR`` differs.
    """

    def __init__(self, downloader=None, *, save_file: bool = True,
                 square: bool = True, filename: str = 'cover',
                 image_format: str = 'jpg', embed: bool = True):
        super().__init__(downloader)
        self.save_file = save_file
        self.square = square
        # Whether an EmbedThumbnail step follows us in this pipeline. Without
        # one, the cropped copy has no consumer and must not be left behind.
        self.embed = embed
        self.filename = _sanitize_cover_stem(filename)
        self.image_format = 'png' if (image_format or '').lower().lstrip('.') == 'png' else 'jpg'

    def _read_source(self, thumbnail: dict[str, Any]) -> Optional[bytes]:
        filepath = thumbnail.get('filepath')
        if filepath and os.path.exists(filepath):
            with open(filepath, 'rb') as handle:
                return handle.read()
        url = thumbnail.get('url')
        if not url:
            return None
        try:
            from yt_dlp.networking import Request  # pylint: disable=import-outside-toplevel
        except ImportError:  # pragma: no cover - very old yt-dlp
            from urllib.request import Request  # pylint: disable=import-outside-toplevel
        response = self._downloader.urlopen(Request(url))
        return response.read()

    def _replace_embed_thumbnail(self, info: dict[str, Any], data: bytes, side: int) -> None:
        """Point yt-dlp's EmbedThumbnail at the cropped image we just made."""
        thumbnails = info.get('thumbnails')
        if not isinstance(thumbnails, list) or not thumbnails:
            return
        media_path = info.get('filepath')
        if not isinstance(media_path, str) or not media_path:
            return
        directory = os.path.dirname(media_path) or '.'
        descriptor, temporary = tempfile.mkstemp(
            prefix='.metube-thumbnail-', suffix=f'.{self.image_format}', dir=directory)
        with os.fdopen(descriptor, 'wb') as handle:
            handle.write(data)

        selected = select_cover_source(thumbnails)
        entry = dict(selected) if selected else {}
        entry.pop('filepath', None)
        entry['filepath'] = temporary
        entry['width'] = entry['height'] = side
        # Drop the previously selected entry so EmbedThumbnail (which takes the
        # last entry that has a filepath) picks ours.
        if selected is not None:
            thumbnails[:] = [t for t in thumbnails if t is not selected]
        thumbnails.append(entry)

    def run(self, info: dict[str, Any]):
        try:
            if not (self.save_file or info.get('thumbnails')):
                return [], info
            source = select_cover_source(info.get('thumbnails'))
            if source is None and info.get('thumbnail'):
                source = {'url': info['thumbnail']}
            if source is None:
                self.to_screen('No thumbnail available for a cover')
                return [], info

            raw = self._read_source(source)
            if not raw:
                return [], info
            encoded, width, height = render_cover(
                raw, image_format=self.image_format, square=self.square)

            if self.embed and info.get('thumbnails'):
                self._replace_embed_thumbnail(info, encoded, width)

            if self.save_file:
                final_dir = info.get('__finaldir')
                if not final_dir:
                    final_dir = os.path.dirname(info.get('filepath') or '')
                if final_dir:
                    destination = atomic_write(
                        final_dir, f'{self.filename}.{self.image_format}', encoded)
                    self.to_screen(f'Square cover saved to "{destination}" ({width}x{height})')
                    info['__metube_cover_file'] = destination
        except Exception as exc:  # supplemental work must never fail a download
            self.report_warning(f'Could not create a square cover: {exc}')
        return [], info
