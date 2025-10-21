import pytest
from unittest import mock
import numpy as np

from face_froze import FaceRecognizer


def test_save_new_customer_tmpdir(monkeypatch, tmp_path):
    # Create an instance but prevent camera initialization
    with mock.patch('face_froze.cv2.VideoCapture') as mock_cap:
        mock_cap.return_value = mock.MagicMock(read=lambda: (True, np.zeros((480,640,3), dtype=np.uint8)))
        fr = FaceRecognizer()

    # Create a fake encoding and mock database methods
    fake_encoding = np.zeros((128,), dtype=float)

    # Mock Danial database methods used by save_new_customer
    monkeypatch.setattr(fr, 'known_face_encodings', [])
    monkeypatch.setattr(fr, 'known_face_names', [])

    # Patch the database object to avoid real DB calls
    import Danial
    class DummyDB:
        def fetch_customer_by_faceid(self, fid):
            return None
        def add_customer(self, CustomerID, CustomerFaceID, Name):
            return 12345
    monkeypatch.setattr('face_froze.database', DummyDB())

    # Use temporary FACES_DIR
    monkeypatch.setenv('FACES_DIR', str(tmp_path / 'Faces'))

    # Call save_new_customer and ensure it returns an id string
    new_name = fr.save_new_customer(fake_encoding)
    assert isinstance(new_name, str)
    # The file should exist
    assert (tmp_path / 'Faces' / f"{new_name}.npy").exists()
