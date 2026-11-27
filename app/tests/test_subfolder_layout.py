"""Tests for the configurable Artist/Album layout and the cover wiring."""

from __future__ import annotations

import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yt_dlp

from download_options import DownloadOptions
from ytdl import Download, DownloadInfo, DownloadQueue, _ConfinedYoutubeDL

SUBFOLDER_TEMPLATE = (
    '%(artist,album_artist,uploader|Unknown Artist)s/%(album|Singles)s/'
    '%(track,title)s.%(ext)s'
)

_VIDEO = {
    '_type': 'video',
    'id': 'vid1',
    'title': 'One',
    'url': 'https://example.com/v1',
    'webpage_url': 'https://example.com/v1',
}


@pytest.fixture
def cfg():
    with tempfile.TemporaryDirectory() as tmp:
        downloads = os.path.join(tmp, 'downloads')
        state = os.path.join(tmp, 'state')
        os.makedirs(downloads)
        os.makedirs(state)
        config = MagicMock()
        config.STATE_DIR = state
        config.DOWNLOAD_DIR = downloads
        config.AUDIO_DOWNLOAD_DIR = downloads
        config.TEMP_DIR = downloads
        config.MAX_CONCURRENT_DOWNLOADS = 2
        config.ALLOW_PRIVATE_ADDRESSES = False
        config.YTDL_OPTIONS = {}
        config.YTDL_OPTIONS_PRESETS = {}
        config.CUSTOM_DIRS = True
        config.CREATE_CUSTOM_DIRS = True
        config.CLEAR_COMPLETED_AFTER = 0
        config.DELETE_FILE_ON_TRASHCAN = 'false'
        config.OUTPUT_TEMPLATE = '%(title)s.%(ext)s'
        config.OUTPUT_TEMPLATE_CHAPTER = '%(title)s - %(section_number)02d - %(section_title)s.%(ext)s'
        # Upstream's defaults: a playlist/channel gets its own folder.
        config.OUTPUT_TEMPLATE_PLAYLIST = '%(playlist_title)s/%(title)s.%(ext)s'
        config.OUTPUT_TEMPLATE_CHANNEL = '%(channel)s/%(title)s.%(ext)s'
        config.SUBFOLDER_ENABLED = False
        config.SUBFOLDER_TEMPLATE = SUBFOLDER_TEMPLATE
        config.SUBFOLDER_TEMPLATE_VIDEO = ''
        config.COVER_SQUARE_CROP = True
        config.COVER_SAVE_FILE = True
        config.COVER_FILENAME = 'cover'
        config.COVER_FORMAT = 'jpg'
        yield config


def _fake_extract(payload):
    def extract(self, url, *_args, **_kwargs):
        return payload
    return extract


async def _queue(cfg, options, payload=None, url='https://example.com/watch?v=1'):
    queue = DownloadQueue(cfg, AsyncMock())
    with patch.object(
        DownloadQueue, '_DownloadQueue__extract_info',
        _fake_extract(payload or _VIDEO),
    ):
        result = await queue.add(url, options)
    assert result['status'] == 'ok', result
    # The pending key is the entry's webpage_url, which the fake feed sets to
    # the item URL rather than the URL that was added.
    return next(iter(queue.pending.dict.values()))


def _audio(**kwargs):
    return DownloadOptions(download_type='audio', format='mp3', auto_start=False, **kwargs)


def _video(**kwargs):
    return DownloadOptions(download_type='video', format='mp4', auto_start=False, **kwargs)


@pytest.mark.asyncio
async def test_audio_uses_subfolder_template_when_enabled(cfg):
    cfg.SUBFOLDER_ENABLED = True
    download = await _queue(cfg, _audio())
    assert download.output_template == SUBFOLDER_TEMPLATE


@pytest.mark.asyncio
async def test_audio_keeps_upstream_template_when_disabled(cfg):
    cfg.SUBFOLDER_ENABLED = False
    download = await _queue(cfg, _audio())
    assert download.output_template == cfg.OUTPUT_TEMPLATE


@pytest.mark.asyncio
async def test_video_keeps_upstream_template_without_an_explicit_video_template(cfg):
    cfg.SUBFOLDER_ENABLED = True  # audio-only switch
    download = await _queue(cfg, _video())
    assert download.output_template == cfg.OUTPUT_TEMPLATE


@pytest.mark.asyncio
async def test_video_uses_its_own_template_when_set(cfg):
    cfg.SUBFOLDER_TEMPLATE_VIDEO = '%(uploader)s/%(title)s.%(ext)s'
    download = await _queue(cfg, _video())
    assert download.output_template == cfg.SUBFOLDER_TEMPLATE_VIDEO


@pytest.mark.asyncio
async def test_playlist_audio_items_also_get_the_subfolder_layout(cfg):
    cfg.SUBFOLDER_ENABLED = True
    feed = {
        '_type': 'playlist',
        'id': 'PL1',
        'title': 'My Playlist',
        'webpage_url': 'https://example.com/playlist?list=PL1',
        'entries': [dict(_VIDEO, _type='url')],
    }
    download = await _queue(
        cfg, _audio(), payload=feed, url='https://example.com/playlist?list=PL1')
    # The feed template would put this in "My Playlist/"; the explicit Artist/
    # Album layout must win instead.
    assert download.output_template == SUBFOLDER_TEMPLATE


@pytest.mark.asyncio
async def test_chapter_files_are_placed_in_the_same_subfolder(cfg):
    cfg.SUBFOLDER_ENABLED = True
    download = await _queue(cfg, _audio(split_by_chapters=True))
    directory = SUBFOLDER_TEMPLATE.rpartition('/')[0]
    assert download.output_template_chapter.startswith(directory + '/')


@pytest.mark.asyncio
async def test_upstream_chapter_template_is_unchanged_without_subfolders(cfg):
    download = await _queue(cfg, _audio(split_by_chapters=True))
    assert download.output_template_chapter == cfg.OUTPUT_TEMPLATE_CHAPTER


def test_subfolder_template_renders_fallbacks_for_missing_metadata():
    with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
        assert ydl.evaluate_outtmpl(
            SUBFOLDER_TEMPLATE, {'title': 'A Song', 'ext': 'mp3'}
        ) == 'Unknown Artist/Singles/A Song.mp3'


def test_subfolder_template_renders_artist_album_and_track():
    with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
        assert ydl.evaluate_outtmpl(SUBFOLDER_TEMPLATE, {
            'artist': 'Björk',
            'album': 'Homogenic',
            'track': 'Jóga',
            'title': 'ignored when track exists',
            'ext': 'flac',
        }) == 'Björk/Homogenic/Jóga.flac'


def _prepared_basename(root, title):
    ydl = _ConfinedYoutubeDL(
        params={'quiet': True, 'outtmpl': {'default': '%(title)s.%(ext)s'},
                'paths': {'home': root}},
        allowed_roots=(root,),
    )
    info = {
        'title': title,
        'id': 'x',
        'ext': 'mp3',
        'extractor': 'generic',
        'extractor_key': 'Generic',
        'webpage_url': 'https://example.com/x',
    }
    path = ydl.prepare_filename(info)
    assert os.path.dirname(path) == os.path.realpath(root)
    return os.path.basename(path)


def test_prepare_filename_repairs_encoding_once_yt_dlp_is_done():
    # yt-dlp sanitises Windows-illegal characters itself (to fullwidth
    # lookalikes); this safety net adds the encoding repair on top.
    with tempfile.TemporaryDirectory() as root:
        assert _prepared_basename(root, 'MÃ¼ller') == 'Müller.mp3'


def test_prepare_filename_protects_reserved_windows_names():
    with tempfile.TemporaryDirectory() as root:
        assert _prepared_basename(root, 'CON') == '_CON.mp3'


def _audio_info():
    return DownloadInfo(
        id='v1', title='One', url='https://example.com/v1', quality='best',
        download_type='audio', codec='auto', format='mp3', folder='',
        custom_name_prefix='', error=None, entry=None, playlist_item_limit=0,
        split_by_chapters=False, chapter_template='',
    )


def _make_download(info, cover_options):
    return Download(
        '.', '.', '%(title)s.%(ext)s', '%(title)s.%(ext)s',
        'best', info.format, {'writethumbnail': True}, info,
        cover_options=cover_options,
    )


def test_cover_processor_runs_before_embed_thumbnail():
    download = _make_download(_audio_info(), {
        'save_file': True, 'square': True, 'filename': 'cover', 'format': 'jpg',
    })
    params = {
        'writethumbnail': True,
        'postprocessors': [{'key': 'EmbedThumbnail'}],
    }
    ydl = download._make_youtube_dl(params)
    names = [type(pp).__name__ for pp in ydl._pps['post_process']]
    assert 'SquareCoverArtPP' in names
    assert names.index('SquareCoverArtPP') < names.index('EmbedThumbnailPP')


def test_cover_processor_is_not_added_to_video_downloads():
    info = _audio_info()
    info.download_type = 'video'
    info.format = 'mp4'
    download = _make_download(info, {
        'save_file': True, 'square': True, 'filename': 'cover', 'format': 'jpg',
    })
    ydl = download._make_youtube_dl({'postprocessors': [], 'writethumbnail': True})
    names = [type(pp).__name__ for pp in ydl._pps.get('post_process', [])]
    assert 'SquareCoverArtPP' not in names


def test_cover_processor_can_be_switched_off():
    download = _make_download(_audio_info(), {
        'save_file': False, 'square': False, 'filename': 'cover', 'format': 'jpg',
    })
    ydl = download._make_youtube_dl({'postprocessors': [{'key': 'EmbedThumbnail'}]})
    names = [type(pp).__name__ for pp in ydl._pps.get('post_process', [])]
    assert 'SquareCoverArtPP' not in names
