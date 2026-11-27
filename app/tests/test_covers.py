"""Unit tests for square cover rendering and the cover postprocessor."""

from __future__ import annotations

import io
import os

import pytest
import yt_dlp
from PIL import Image

from covers import (
    SquareCoverArtPP,
    center_square,
    render_cover,
    select_cover_source,
)


def _image(width: int, height: int, color=(255, 0, 0)) -> Image.Image:
    return Image.new('RGB', (width, height), color)


def _encode(image: Image.Image, fmt: str = 'PNG') -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, fmt)
    return buffer.getvalue()


def _decode(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert('RGB')


@pytest.mark.parametrize('size', [(160, 90), (120, 80), (90, 160), (64, 64), (100, 300)])
def test_centre_crop_is_square_and_never_upscales(size):
    width, height = size
    cropped = center_square(_image(width, height))
    assert cropped.size == (min(width, height), min(width, height))


def test_centre_crop_keeps_the_middle_and_discards_edges():
    # Three vertical bands: red | green | blue.
    image = Image.new('RGB', (150, 50), (0, 255, 0))
    image.paste((255, 0, 0), (0, 0, 50, 50))
    image.paste((0, 0, 255), (100, 0, 150, 50))
    cropped = center_square(image)
    assert cropped.size == (50, 50)
    assert cropped.getpixel((25, 25)) == (0, 255, 0)


@pytest.mark.parametrize('size', [(160, 90), (120, 80), (90, 160), (64, 64)])
def test_render_cover_output_is_one_to_one_jpeg(size):
    data, width, height = render_cover(_encode(_image(*size)), image_format='jpg')
    assert width == height == min(size)
    decoded = _decode(data)
    assert decoded.size == (min(size), min(size))
    assert decoded.format is None  # decoded via Pillow after load
    assert data[:2] == b'\xff\xd8'  # JPEG SOI


def test_render_cover_png():
    data, width, height = render_cover(_encode(_image(160, 90)), image_format='png')
    assert (width, height) == (90, 90)
    assert data[:8] == b'\x89PNG\r\n\x1a\n'


def test_render_cover_accepts_webp_input_and_does_not_stretch():
    data, width, height = render_cover(_encode(_image(200, 100), 'WEBP'))
    assert (width, height) == (100, 100)


def test_render_cover_flattens_transparency():
    image = Image.new('RGBA', (80, 40), (255, 0, 0, 128))
    data, width, height = render_cover(_encode(image))
    assert width == height == 40
    assert _decode(data).getpixel((20, 20))[0] > 200


def test_square_can_be_disabled():
    data, width, height = render_cover(_encode(_image(160, 90)), square=False)
    assert (width, height) == (160, 90)


class TestPickSource:
    def test_prefers_the_largest_square_over_a_larger_widescreen(self):
        thumbnails = [
            {'url': 'wide', 'width': 1920, 'height': 1080},
            {'url': 'small-square', 'width': 400, 'height': 400},
            {'url': 'large-square', 'width': 800, 'height': 800},
        ]
        assert select_cover_source(thumbnails)['url'] == 'large-square'

    def test_falls_back_to_the_best_available_thumbnail(self):
        thumbnails = [
            {'url': 'small', 'width': 320, 'height': 180},
            {'url': 'large', 'width': 1280, 'height': 720},
        ]
        assert select_cover_source(thumbnails)['url'] == 'large'

    def test_ignores_entries_without_a_source(self):
        thumbnails = [{'width': 5000, 'height': 5000}, {'url': 'ok', 'width': 10, 'height': 10}]
        assert select_cover_source(thumbnails)['url'] == 'ok'

    def test_no_thumbnails(self):
        assert select_cover_source(None) is None
        assert select_cover_source([]) is None


class TestPostProcessor:
    def _run(self, tmp_path, thumbnails, save_file=True, filename='cover', fmt='jpg'):
        audio = tmp_path / 'Artist' / 'Album' / 'Track.mp3'
        audio.parent.mkdir(parents=True)
        audio.write_bytes(b'audio')
        ydl = yt_dlp.YoutubeDL({'quiet': True})
        processor = SquareCoverArtPP(
            ydl, save_file=save_file, filename=filename, image_format=fmt)
        info = {
            'filepath': str(audio),
            '__finaldir': str(audio.parent),
            'thumbnails': thumbnails,
        }
        return processor.run(info)[1], audio.parent

    def test_writes_a_square_cover_next_to_the_media(self, tmp_path):
        thumb = tmp_path / 'thumb.webp'
        thumb.write_bytes(_encode(_image(160, 90)))
        info, directory = self._run(tmp_path, [{'filepath': str(thumb), 'url': 'x'}])
        cover = directory / 'cover.jpg'
        assert cover.exists()
        assert Image.open(cover).size == (90, 90)
        assert info['__metube_cover_file'] == str(cover)

    def test_points_embedding_at_the_cropped_image(self, tmp_path):
        thumb = tmp_path / 'thumb.png'
        thumb.write_bytes(_encode(_image(160, 90)))
        info, _ = self._run(tmp_path, [{'filepath': str(thumb), 'url': 'x'}])
        entry = info['thumbnails'][-1]
        assert entry['width'] == entry['height'] == 90
        assert os.path.exists(entry['filepath'])
        assert Image.open(entry['filepath']).size == (90, 90)

    def test_custom_filename_and_png(self, tmp_path):
        thumb = tmp_path / 'thumb.png'
        thumb.write_bytes(_encode(_image(50, 50)))
        _, directory = self._run(
            tmp_path, [{'filepath': str(thumb)}], filename='folder', fmt='png')
        assert (directory / 'folder.png').exists()

    def test_filename_cannot_escape_the_directory(self, tmp_path):
        thumb = tmp_path / 'thumb.png'
        thumb.write_bytes(_encode(_image(50, 50)))
        _, directory = self._run(tmp_path, [{'filepath': str(thumb)}], filename='../../evil')
        assert not (tmp_path / 'evil.jpg').exists()
        assert (directory / 'evil.jpg').exists()

    def test_save_file_can_be_disabled(self, tmp_path):
        thumb = tmp_path / 'thumb.png'
        thumb.write_bytes(_encode(_image(50, 50)))
        _, directory = self._run(tmp_path, [{'filepath': str(thumb)}], save_file=False)
        assert not (directory / 'cover.jpg').exists()

    def test_without_an_embed_step_no_thumbnail_copy_is_left_behind(self, tmp_path):
        thumb = tmp_path / 'thumb.png'
        thumb.write_bytes(_encode(_image(50, 50)))
        ydl = yt_dlp.YoutubeDL({'quiet': True})
        processor = SquareCoverArtPP(ydl, save_file=True, embed=False)
        audio = tmp_path / 'Track.mp3'
        audio.write_bytes(b'audio')
        info = {'filepath': str(audio), '__finaldir': str(tmp_path),
                'thumbnails': [{'filepath': str(thumb)}]}
        info = processor.run(info)[1]
        assert (tmp_path / 'cover.jpg').exists()
        assert not [p for p in os.listdir(tmp_path) if p.startswith('.metube-thumbnail-')]
        assert info['thumbnails'][0]['filepath'] == str(thumb)

    def test_bad_image_warns_and_does_not_raise(self, tmp_path):
        thumb = tmp_path / 'thumb.jpg'
        thumb.write_bytes(b'not an image')
        info, directory = self._run(tmp_path, [{'filepath': str(thumb)}])
        assert not (directory / 'cover.jpg').exists()
        assert info['filepath'].endswith('Track.mp3')
