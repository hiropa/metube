"""Unit tests for path_sanitizer.sanitize_path_component."""

from __future__ import annotations

import unicodedata

import pytest

from path_sanitizer import configure, sanitize_path_component


@pytest.fixture(autouse=True)
def _reset_settings():
    configure(fix_encoding=True, replacement='_', ascii_only=False)
    yield
    configure(fix_encoding=True, replacement='_', ascii_only=False)


class TestEncodingRepair:
    def test_repairs_latin1_mojibake(self):
        assert sanitize_path_component('MÃ¼ller') == 'Müller'
        assert sanitize_path_component('CafÃ©') == 'Café'

    def test_repairs_cp1252_smart_quotes(self):
        assert sanitize_path_component('Donâ€™t Stop') == 'Don’t Stop'
        assert sanitize_path_component('Ã‰tÃ©') == 'Été'

    def test_repaired_text_is_nfc(self):
        repaired = sanitize_path_component('CafeÌ\x81')
        assert repaired == unicodedata.normalize('NFC', repaired)

    def test_korean_is_preserved(self):
        assert sanitize_path_component('아이유 - 밤편지') == '아이유 - 밤편지'

    def test_japanese_is_preserved(self):
        assert sanitize_path_component('米津玄師 - Lemon') == '米津玄師 - Lemon'

    def test_chinese_cyrillic_and_arabic_are_preserved(self):
        assert sanitize_path_component('周杰伦 - 稻香') == '周杰伦 - 稻香'
        assert sanitize_path_component('Кино - Звезда') == 'Кино - Звезда'
        assert sanitize_path_component('فيروز - نسم علينا الهوا') == 'فيروز - نسم علينا الهوا'

    def test_correctly_encoded_umlauts_are_preserved(self):
        assert sanitize_path_component('Björk') == 'Björk'
        assert sanitize_path_component('Sigur Rós') == 'Sigur Rós'

    def test_unrepairable_replacement_char_becomes_fallback(self):
        assert sanitize_path_component('bad\ufffdname') == 'bad_name'

    def test_fix_encoding_can_be_disabled(self):
        assert sanitize_path_component('MÃ¼ller', fix_encoding=False) == 'MÃ¼ller'


class TestFilesystemSafety:
    def test_replaces_windows_invalid_characters(self):
        assert sanitize_path_component('a:b*c?d"e<f>g|h') == 'a_b_c_d_e_f_g_h'

    def test_replaces_control_characters(self):
        assert sanitize_path_component('a\x00b\x1fc') == 'a_b_c'
        assert sanitize_path_component('a\x85b') == 'a_b'

    def test_path_separators_are_never_kept(self):
        assert sanitize_path_component('AC/DC') == 'AC_DC'
        assert sanitize_path_component('a\\b') == 'a_b'

    def test_path_traversal_is_removed(self):
        result = sanitize_path_component('../../../../etc/passwd')
        assert '..' not in result
        assert '/' not in result and '\\' not in result
        assert result == 'etc_passwd'

    def test_absolute_paths_cannot_escape(self):
        assert sanitize_path_component('/tmp/x') == '_tmp_x'
        assert sanitize_path_component('C:\\Windows') == 'C__Windows'
        assert sanitize_path_component('..') == '_'

    def test_legitimate_ellipsis_is_preserved(self):
        assert sanitize_path_component('Wait... what?') == 'Wait... what_'

    def test_trailing_dots_and_spaces_are_stripped(self):
        assert sanitize_path_component('name. ') == 'name'
        assert sanitize_path_component(' name ') == 'name'

    def test_dot_only_components_are_removed(self):
        assert sanitize_path_component('.') == '_'
        assert sanitize_path_component('..') == '_'
        assert sanitize_path_component('...') == '_'

    def test_leading_dot_on_a_real_name_is_kept(self):
        # Operator-chosen folders such as '.metube-videos' must survive.
        assert sanitize_path_component('.hidden') == '.hidden'

    def test_empty_and_whitespace_become_underscore(self):
        assert sanitize_path_component('') == '_'
        assert sanitize_path_component('   ') == '_'
        assert sanitize_path_component('...') == '_'

    @pytest.mark.parametrize('reserved', ['CON', 'con', 'PRN', 'NUL', 'COM1', 'com9', 'LPT1', 'lpt9'])
    def test_reserved_windows_names_are_prefixed(self, reserved):
        assert sanitize_path_component(reserved) == f'_{reserved}'

    def test_reserved_name_with_extension_is_prefixed(self):
        assert sanitize_path_component('con.txt') == '_con.txt'

    def test_non_reserved_lookalike_is_untouched(self):
        assert sanitize_path_component('console') == 'console'
        assert sanitize_path_component('COM10') == 'COM10'

    def test_custom_replacement_character(self):
        assert sanitize_path_component('a:b', replacement='-') == 'a-b'

    def test_empty_replacement_removes_illegal_characters(self):
        assert sanitize_path_component('a:b', replacement='') == 'ab'

    def test_non_string_passthrough(self):
        assert sanitize_path_component(None) is None
        assert sanitize_path_component(42) == 42


class TestLength:
    def test_ascii_name_is_truncated_to_255_bytes_and_keeps_extension(self):
        result = sanitize_path_component('a' * 400 + '.mp3')
        assert result.endswith('.mp3')
        assert len(result.encode('utf-8')) <= 255

    def test_multibyte_name_is_truncated_on_a_character_boundary(self):
        result = sanitize_path_component('한' * 200 + '.flac')
        assert len(result.encode('utf-8')) <= 255
        assert result.endswith('.flac')
        # No U+FFFD: the cut landed between characters, not inside one.
        assert '\ufffd' not in result
        result.encode('utf-8')  # decodable/encondable

    def test_short_name_is_untouched(self):
        assert sanitize_path_component('short name.mp3') == 'short name.mp3'


class TestAsciiOnly:
    def test_transliterates_accented_latin(self):
        assert sanitize_path_component('Beyoncé', ascii_only=True) == 'Beyonce'

    def test_non_latin_becomes_fallback(self):
        assert sanitize_path_component('안녕', ascii_only=True) == '_'

    def test_configure_applies_ascii_only(self):
        configure(ascii_only=True)
        assert sanitize_path_component('Björk') == 'Bjork'
