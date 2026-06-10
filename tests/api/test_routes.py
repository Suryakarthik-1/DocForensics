import io
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def make_jpg_bytes():
    img = np.ones((64, 64, 3), dtype=np.uint8) * 200
    _, buf = cv2.imencode('.jpg', img)
    return buf.tobytes()


def test_health_returns_ok():
    resp = client.get('/api/health')
    assert resp.status_code == 200
    assert resp.json()['status'] == 'ok'


def test_analyze_returns_200():
    resp = client.post(
        '/api/analyze',
        files={'file': ('test.jpg', io.BytesIO(make_jpg_bytes()), 'image/jpeg')},
    )
    assert resp.status_code == 200


def test_analyze_response_has_required_fields():
    resp = client.post(
        '/api/analyze',
        files={'file': ('test.jpg', io.BytesIO(make_jpg_bytes()), 'image/jpeg')},
    )
    body = resp.json()
    assert 'is_tampered' in body
    assert 'confidence' in body
    assert 'evidence' in body
    assert 'per_detector' in body
    assert 'heatmap_base64' in body


def test_analyze_confidence_in_range():
    resp = client.post(
        '/api/analyze',
        files={'file': ('test.jpg', io.BytesIO(make_jpg_bytes()), 'image/jpeg')},
    )
    conf = resp.json()['confidence']
    assert 0.0 <= conf <= 1.0


def test_analyze_rejects_unsupported_file():
    resp = client.post(
        '/api/analyze',
        files={'file': ('virus.exe', io.BytesIO(b'MZ'), 'application/octet-stream')},
    )
    assert resp.status_code == 400


def test_per_detector_list_not_empty():
    resp = client.post(
        '/api/analyze',
        files={'file': ('test.jpg', io.BytesIO(make_jpg_bytes()), 'image/jpeg')},
    )
    assert len(resp.json()['per_detector']) > 0