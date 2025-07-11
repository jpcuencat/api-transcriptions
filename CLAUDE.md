# Video Transcription API - Architecture & Coding Standards

## Descripción del Proyecto

API de transcripción de videos desarrollada con **FastAPI** que utiliza el modelo **Whisper de OpenAI** para convertir automáticamente videos a texto y generar archivos de subtítulos SRT. La API soporta múltiples idiomas, evaluación de calidad y capacidades de traducción.

## Arquitectura del Sistema

### Patrón Arquitectónico
- **Arquitectura en Capas**: API → Servicios → Utilidades
- **Separación de Responsabilidades**: Endpoints, lógica de negocio, modelos de datos
- **Procesamiento Asíncrono**: Background tasks con FastAPI
- **Configuración Centralizada**: Pydantic Settings con soporte .env

### Flujo de Procesamiento
```
1. Upload & Validación → 2. Creación de Job → 3. Procesamiento Background → 4. Almacenamiento de Resultados
```

## Estructura de Directorios

```
app/
├── main.py                    # Punto de entrada de la aplicación
├── api/endpoints/             # Definiciones de endpoints REST
│   ├── health.py             # Endpoints de health check
│   └── transcription.py      # API principal de transcripción
├── core/                     # Componentes centrales
│   ├── config.py            # Gestión de configuración
│   └── security.py          # Autenticación y seguridad
├── models/                   # Modelos de datos
│   └── schemas.py           # Modelos Pydantic para API
├── services/                 # Lógica de negocio
│   ├── audio_extractor.py      # Conversión video → audio
│   ├── transcription_service.py # Integración con Whisper
│   ├── quality_evaluator.py    # Evaluación de calidad
│   ├── subtitle_generator.py   # Generación de archivos SRT
│   └── translation_service.py  # Servicios de traducción
└── utils/                    # Funciones de utilidad
    ├── file_handler.py      # Gestión de archivos
    └── validators.py        # Validación de entrada
```

### Propósito de Cada Módulo

- **`api/endpoints/`**: Definiciones de rutas REST, manejo de requests/responses
- **`core/`**: Configuración global, seguridad, autenticación
- **`models/`**: Esquemas Pydantic para validación y serialización de datos
- **`services/`**: Lógica de negocio independiente, servicios especializados
- **`utils/`**: Funciones de apoyo, validadores, manejadores de archivos

## Servicios Principales

### 1. TranscriptionService
- **Responsabilidad**: Gestión de modelos Whisper y transcripción de audio
- **Características**: Carga dinámica de modelos, optimización GPU/CPU, timestamps
- **Tecnologías**: OpenAI Whisper, PyTorch, langdetect

### 2. AudioExtractor  
- **Responsabilidad**: Procesamiento de video y extracción de audio
- **Características**: Soporte multi-formato, extracción optimizada (16kHz mono)
- **Tecnologías**: FFmpeg-python

### 3. QualityEvaluator
- **Responsabilidad**: Evaluación de calidad de transcripciones
- **Métricas**: Confidence score, speech rate, silence ratio, repetition score
- **Niveles**: Excelente (≥0.95), Bueno (≥0.85), Aceptable (≥0.70), Pobre (<0.70)

### 4. SubtitleGenerator
- **Responsabilidad**: Generación optimizada de archivos SRT
- **Características**: Segmentación de texto, ajuste de duración, formato SRT

### 5. TranslationService
- **Responsabilidad**: Traducción multi-idioma
- **Características**: Traducción por segmentos, 12 idiomas soportados
- **Tecnologías**: deep-translator, Google Translate API

## Convenciones de Codificación

### Estilo de Código (PEP 8)

#### Imports
```python
# Standard library imports
import os
import time
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any

# Third-party imports  
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# Local imports
from app.core.config import settings
from app.models.schemas import TranscriptionRequest
```

#### Nomenclatura
```python
# Classes: PascalCase
class TranscriptionService:
    pass

# Functions and variables: snake_case
def process_transcription(file_path: str) -> TranscriptionResult:
    job_id = str(uuid.uuid4())
    return result

# Constants: UPPER_SNAKE_CASE
MAX_FILE_SIZE_MB = 500
SUPPORTED_FORMATS = [".mp4", ".avi", ".mov"]
```

#### Espaciado y Líneas
- **Líneas máximo**: 100 caracteres
- **Indentación**: 4 espacios (no tabs)
- **Líneas en blanco**: 2 entre clases, 1 entre métodos
- **Espacios**: Alrededor de operadores y después de comas

### Docstrings

#### Funciones y Métodos
```python
def extract_audio(self, video_path: str, output_path: str) -> VideoInfo:
    """
    Extrae audio de un archivo de video y obtiene metadata.
    
    Args:
        video_path (str): Ruta al archivo de video de entrada
        output_path (str): Ruta donde guardar el archivo de audio extraído
        
    Returns:
        VideoInfo: Información del video incluyendo duración, tamaño y códecs
        
    Raises:
        AudioExtractionError: Si la extracción de audio falla
        FileNotFoundError: Si el archivo de video no existe
    """
```

#### Clases
```python
class TranscriptionService:
    """
    Servicio para la transcripción de audio utilizando modelos Whisper.
    
    Este servicio maneja la carga de modelos, transcripción de audio,
    y optimizaciones para diferentes tamaños de modelo.
    
    Attributes:
        current_model: Modelo Whisper cargado actualmente
        model_size: Tamaño del modelo en uso (tiny, base, small, medium, large)
        cache_dir: Directorio para caché de modelos
    """
```

#### Módulos
```python
"""
Servicio de transcripción de audio usando Whisper.

Este módulo proporciona funcionalidades para:
- Carga y gestión de modelos Whisper
- Transcripción de archivos de audio
- Optimización de parámetros para precisión
- Cálculo de scores de confianza

Ejemplo de uso:
    service = TranscriptionService()
    result = await service.transcribe_audio("audio.wav", "base", "es")
"""
```

### Type Hints

#### Tipos Básicos
```python
from typing import Optional, List, Dict, Any, Union

def process_file(
    file_path: str,
    options: Optional[Dict[str, Any]] = None,
    languages: List[str] = None
) -> TranscriptionResult:
    pass
```

#### Tipos Personalizados
```python
from typing import TypeVar, Generic

JobId = str
Language = str
ModelSize = str

JobStorage = Dict[JobId, TranscriptionResult]
```

### Manejo de Errores

#### Excepciones Específicas
```python
class TranscriptionError(Exception):
    """Error durante el proceso de transcripción."""
    pass

class AudioExtractionError(Exception):
    """Error durante la extracción de audio."""
    pass
```

#### Manejo con Try-Catch
```python
try:
    result = await transcription_service.transcribe(audio_path, model_size)
    logger.info(f"Transcripción completada para job {job_id}")
except TranscriptionError as e:
    logger.error(f"Error en transcripción {job_id}: {str(e)}")
    raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
except Exception as e:
    logger.error(f"Error inesperado en {job_id}: {str(e)}")
    raise HTTPException(status_code=500, detail="Internal server error")
```

### Logging

#### Configuración
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("transcription_api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
```

#### Uso en Código
```python
logger.info(f"Iniciando transcripción para archivo: {filename}")
logger.warning(f"Calidad baja detectada en transcripción {job_id}: {quality_score}")
logger.error(f"Fallo en extracción de audio: {str(e)}")
```

### Configuración y Validación

#### Pydantic Models
```python
class TranscriptionRequest(BaseModel):
    """Modelo para requests de transcripción."""
    language: Optional[str] = Field(default="auto", description="Código de idioma")
    model_size: Optional[str] = Field(default="base", description="Tamaño del modelo")
    
    class Config:
        json_schema_extra = {
            "example": {
                "language": "es",
                "model_size": "base"
            }
        }
```

#### Settings Management
```python
class Settings(BaseSettings):
    """Configuración de la aplicación."""
    API_TITLE: str = "Video Transcription API"
    DEBUG: bool = True
    MAX_FILE_SIZE_MB: int = 500
    
    class Config:
        env_file = ".env"
```

## Comandos de Desarrollo

### Instalación
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### Ejecución
```bash
# Desarrollo
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Producción
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Testing
```bash
# Ejecutar tests (cuando estén implementados)
pytest tests/

# Coverage
pytest --cov=app tests/
```

### Linting y Formato
```bash
# Formato de código
black app/
isort app/

# Linting
flake8 app/
pylint app/

# Type checking
mypy app/
```

## Buenas Prácticas

### Seguridad
- Validación de archivos subidos (tamaño, tipo, extensión)
- Autenticación con API keys
- Rate limiting por IP
- Sanitización de nombres de archivo
- Limpieza automática de archivos temporales

### Performance
- Carga perezosa de modelos Whisper
- Procesamiento asíncrono con background tasks
- Cacheo de modelos en disco
- Optimización de parámetros Whisper para velocidad/precisión

### Mantenibilidad
- Separación clara de responsabilidades
- Inyección de dependencias con FastAPI
- Configuración centralizada
- Logging comprehensivo
- Limpieza automática de recursos

### Escalabilidad
- Arquitectura de servicios independientes
- Procesamiento background asíncrono
- Gestión de estado sin dependencias de memoria
- Configuración flexible por ambiente

## Dependencias Principales

```
fastapi==0.104.1          # Framework web asíncrono
uvicorn[standard]==0.24.0 # Servidor ASGI
openai-whisper==20231117  # Modelo de transcripción
ffmpeg-python==0.2.0      # Procesamiento de video
pydantic==2.5.0           # Validación de datos
aiofiles==23.2.1          # Operaciones de archivo asíncronas
deep-translator==1.11.4   # Servicios de traducción
```

## Consideraciones para Producción

### Mejoras Recomendadas
- **Base de Datos**: PostgreSQL para persistencia de jobs
- **Queue System**: Redis/Celery para procesamiento background
- **Autenticación**: JWT tokens, OAuth2
- **Almacenamiento**: S3/MinIO para archivos
- **Monitoreo**: Prometheus, Grafana
- **Containerización**: Docker, Kubernetes
- **CI/CD**: GitHub Actions, testing automatizado