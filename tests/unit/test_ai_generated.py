import numpy as np

from detectors.ai_generated import frequency_analysis, check_exif_absence


def test_frequency_score_in_range(clean_image):
    score = frequency_analysis(clean_image)
    assert 0.0 <= score <= 1.0


def test_frequency_score_is_float(tampered_image):
    score = frequency_analysis(tampered_image)
    assert isinstance(score, float)


def test_exif_absence_on_missing_file():
    # a non-existent path must fail safe to (0.0, '')
    score, msg = check_exif_absence('/tmp/does_not_exist_xyz.jpg')
    assert score == 0.0
    assert msg == ''


def test_exif_absence_on_image_without_camera_tags(tmp_path):
    import cv2
    p = tmp_path / 'plain.jpg'
    cv2.imwrite(str(p), (np.ones((32, 32, 3)) * 128).astype(np.uint8))
    score, msg = check_exif_absence(str(p))
    # synthetic jpg has no camera EXIF → mild suspicion
    assert score >= 0.0
    assert isinstance(msg, str)
