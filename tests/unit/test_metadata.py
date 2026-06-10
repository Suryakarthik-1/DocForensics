import pytest

from detectors.metadata import metadata_red_flags


def test_photoshop_flagged():
    score, flags = metadata_red_flags({'producer': 'Adobe Photoshop 2024'})
    assert score > 0.0
    assert len(flags) > 0


def test_clean_metadata_no_flags():
    score, flags = metadata_red_flags({'producer': 'Acrobat Distiller'})
    assert score == 0.0
    assert flags == []


def test_date_mismatch_flagged():
    score, flags = metadata_red_flags({
        'creation_date': '2024-01-01',
        'mod_date': '2024-06-15'
    })
    assert score > 0.0


def test_pdf_incremental_update_flagged():
    score, flags = metadata_red_flags({'incremental_updates': True})
    assert score > 0.0


def test_score_never_exceeds_1():
    score, _ = metadata_red_flags({
        'producer': 'Adobe Photoshop',
        'incremental_updates': True,
        'creation_date': '2023-01-01',
        'mod_date': '2024-06-01',
    })
    assert score <= 1.0