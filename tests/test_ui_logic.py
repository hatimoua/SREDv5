import pytest
from unittest.mock import MagicMock, patch
from sred.ui.validation import validate_schema, validate_data_dir, validate_db_connection
from sred.storage.files import sanitize_filename, compute_sha256, save_upload
from sred.models.core import Run, Person, File
from pathlib import Path

def test_sanitization():
    assert sanitize_filename("foo bar.txt") == "foo_bar.txt"
    assert sanitize_filename("../foo.txt") == ".._foo.txt"
    assert sanitize_filename("foo/bar") == "foo_bar"

def test_hashing():
    data = b"hello world"
    expected = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
    assert compute_sha256(data) == expected

def test_save_upload(tmp_path):
    # Mock DATA_DIR
    with patch("sred.storage.files.DATA_DIR", tmp_path):
        mock_file = MagicMock()
        mock_file.getvalue.return_value = b"test content"
        mock_file.name = "test.txt"
        mock_file.type = "text/plain"
        
        run_id = 999
        path, sha, size, mime = save_upload(run_id, mock_file)
        
        expected_path = tmp_path / "runs" / "999" / "uploads" / f"{sha}_test.txt"
        assert expected_path.exists()
        assert size == 12
        assert mime == "text/plain"
        # Check return path is relative
        assert path == f"runs/999/uploads/{sha}_test.txt"

def test_validation():
    # Schema validation should pass if models are correct
    errors = validate_schema()
    assert not errors
    
    # DB connection check might fail if no DB, but logic runs
    # We can mock session to test success path
    with patch("sred.ui.validation.Session") as mock_session:
        errors = validate_db_connection()
        assert not errors
