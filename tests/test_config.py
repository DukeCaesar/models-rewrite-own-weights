import pytest
from self_rewrite.config import preset


def test_reported_session_settings():
    for size in ('2B','4B','9B','27B'):
        assert preset(size).max_tokens == 16384
        assert preset(size,repair=True).diagnosis_count == 64
        assert preset(size,repair=True).hours == 2.5
    assert preset(short=True).max_tokens == 4096
    second = preset(protocol=2)
    assert (second.reserved_count,second.check_count,second.max_generation_answers) == (300,256,128)
    with pytest.raises(ValueError): preset('9B',protocol=2)
