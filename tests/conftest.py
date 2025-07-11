"""
Configuración global de pytest para las pruebas de la aplicación.

Este archivo contiene fixtures compartidas y configuración global
para todas las pruebas del proyecto.
"""

import pytest
import tempfile
import shutil
import os
from pathlib import Path
from unittest.mock import Mock
from fastapi.testclient import TestClient

# Importar la aplicación
from app.main import app
from app.core.config import settings


@pytest.fixture(scope="session")
def test_settings():
    """
    Configuración de pruebas que sobrescribe las configuraciones por defecto.
    """
    original_debug = settings.DEBUG
    original_temp_dir = settings.TEMP_DIR
    
    # Configurar para pruebas
    settings.DEBUG = True
    settings.TEMP_DIR = tempfile.mkdtemp()
    
    yield settings
    
    # Restaurar configuración original
    settings.DEBUG = original_debug
    settings.TEMP_DIR = original_temp_dir


@pytest.fixture(scope="session")
def test_app():
    """
    Instancia de la aplicación FastAPI para pruebas.
    """
    return app


@pytest.fixture
def client(test_app):
    """
    Cliente de prueba HTTP para la aplicación.
    """
    return TestClient(test_app)


@pytest.fixture
def temp_dir():
    """
    Directorio temporal que se limpia automáticamente después de cada prueba.
    """
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def test_video_path():
    """
    Ruta al video de prueba. Verifica que existe antes de usarlo.
    """
    video_path = "/mnt/d/api-transcriptions/media/AC201 12 Data Loading (1080p).mp4"
    
    if os.path.exists(video_path):
        return video_path
    else:
        # Buscar archivos de video alternativos en el directorio media
        media_dir = "/mnt/d/api-transcriptions/media"
        if os.path.exists(media_dir):
            for file in os.listdir(media_dir):
                if file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    return os.path.join(media_dir, file)
        
        # Si no se encuentra ningún video, retornar None
        return None


@pytest.fixture
def valid_api_key():
    """
    API key válida para pruebas de autenticación.
    """
    return "dev_api_key_12345"


@pytest.fixture
def invalid_api_key():
    """
    API key inválida para pruebas de autenticación.
    """
    return "invalid_key_123"


@pytest.fixture
def auth_headers(valid_api_key):
    """
    Headers de autenticación válidos.
    """
    return {"Authorization": f"Bearer {valid_api_key}"}


@pytest.fixture
def mock_transcription_result():
    """
    Resultado de transcripción mock para pruebas.
    """
    return {
        'text': 'This is a mock transcription for testing purposes.',
        'segments': [
            {
                'id': 0,
                'start': 0.0,
                'end': 3.0,
                'text': 'This is a mock transcription',
                'confidence': 0.95
            },
            {
                'id': 1,
                'start': 3.0,
                'end': 6.0,
                'text': ' for testing purposes.',
                'confidence': 0.88
            }
        ],
        'language': 'en'
    }


@pytest.fixture
def mock_video_info():
    """
    Información de video mock para pruebas.
    """
    from app.models.schemas import VideoInfo
    return VideoInfo(
        duration=120.5,
        size=15728640,  # 15MB
        video_codec="h264",
        audio_codec="aac",
        fps=30.0
    )


@pytest.fixture
def sample_video_segments():
    """
    Segmentos de video de ejemplo para pruebas de subtítulos.
    """
    return [
        {
            'id': 0,
            'start': 0.0,
            'end': 2.5,
            'text': 'Welcome to this video tutorial.'
        },
        {
            'id': 1,
            'start': 2.5,
            'end': 5.0,
            'text': 'In this session we will learn about data loading.'
        },
        {
            'id': 2,
            'start': 5.0,
            'end': 8.5,
            'text': 'Data loading is a crucial step in any data processing pipeline.'
        },
        {
            'id': 3,
            'start': 8.5,
            'end': 12.0,
            'text': 'Let\'s start by understanding the basic concepts.'
        }
    ]


def pytest_configure(config):
    """
    Configuración personalizada de pytest.
    """
    # Añadir marcadores personalizados
    config.addinivalue_line(
        "markers", "integration: marca las pruebas como pruebas de integración"
    )
    config.addinivalue_line(
        "markers", "slow: marca las pruebas que toman mucho tiempo"
    )
    config.addinivalue_line(
        "markers", "api: marca las pruebas de endpoints API"
    )
    config.addinivalue_line(
        "markers", "unit: marca las pruebas unitarias"
    )


def pytest_collection_modifyitems(config, items):
    """
    Modifica la colección de pruebas para añadir marcadores automáticamente.
    """
    for item in items:
        # Añadir marcador 'slow' a pruebas que contienen 'real_video' en el nombre
        if "real_video" in item.name:
            item.add_marker(pytest.mark.slow)
        
        # Añadir marcador 'integration' a pruebas de integración
        if "integration" in item.name or "complete_flow" in item.name:
            item.add_marker(pytest.mark.integration)
        
        # Añadir marcador 'api' a pruebas de endpoints
        if "api_endpoint" in item.name or "test_endpoint" in item.name:
            item.add_marker(pytest.mark.api)
        
        # Añadir marcador 'unit' a pruebas unitarias básicas
        if any(word in item.name for word in ["validator", "service", "handler"]):
            item.add_marker(pytest.mark.unit)