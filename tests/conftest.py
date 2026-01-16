"""Pytest configuration and fixtures for FastTTS tests."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client for the FastTTS API."""
    from fasttts.app import app
    return TestClient(app)
