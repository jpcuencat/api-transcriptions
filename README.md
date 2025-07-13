# 🎥 API Transcriptions

Una API robusta desarrollada en **FastAPI** para transcripción automática de videos, traducción de contenido y generación de subtítulos usando **Whisper** de OpenAI y modelos de traducción **Helsinki-NLP**.

## ✨ Características Principales

- 🎯 **Transcripción automática** de videos usando Whisper (OpenAI)
- 🌍 **Traducción multiidioma** con modelos locales Helsinki-NLP
- 📝 **Generación de subtítulos SRT** con tiempos precisos de Whisper
- 📊 **Evaluación de calidad** automática de transcripciones
- 🔄 **Procesamiento asíncrono** de trabajos en background
- 🛡️ **Autenticación por API Key** y rate limiting
- 📁 **Soporte múltiples formatos** de video (.mp4, .avi, .mov, .mkv, .webm)

## 🚀 Instalación Rápida

### Prerrequisitos
- Python 3.8+
- FFmpeg instalado en el sistema

### 1. Clonar el repositorio
```bash
git clone https://github.com/jpcuencat/api-transcriptions.git
cd api-transcriptions
```

### 2. Crear entorno virtual
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Ejecutar la aplicación
```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

La API estará disponible en: `http://localhost:8000`

## 📖 Documentación de la API

### Endpoints Principales

#### 🎬 Transcribir Video
```http
POST /api/v1/transcribe
```

**Parámetros:**
- `video_file`: Archivo de video (form-data)
- `language`: Código de idioma o 'auto' (opcional, default: 'auto')
- `model_size`: Tamaño del modelo Whisper (opcional, default: 'base')
- `translate_to`: Idioma destino para traducción (opcional)
- `quality_evaluation`: Habilitar evaluación de calidad (opcional, default: true)

**Headers:**
```
X-API-Key: dev_api_key_12345
Content-Type: multipart/form-data
```

**Respuesta:**
```json
{
  "job_id": "uuid-string",
  "status": "processing",
  "created_at": "2025-07-13T17:00:00"
}
```

#### 📊 Estado del Trabajo
```http
GET /api/v1/transcribe/{job_id}/status
```

**Respuesta exitosa:**
```json
{
  "job_id": "uuid-string",
  "status": "completed",
  "transcription_text": "Texto transcrito...",
  "translation_text": "Texto traducido...",
  "detected_language": "en",
  "segments_count": 25,
  "translation_segments_count": 25,
  "srt_available": true,
  "quality_score": 0.95,
  "processing_time": 45.6
}
```

#### ⬇️ Descargar Subtítulos
```http
GET /api/v1/transcribe/{job_id}/download
```

#### 🌐 Idiomas Soportados
```http
GET /api/v1/languages
```

**Respuesta:**
```json
{
  "transcription_languages": {
    "auto": "Auto-detect",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian"
  },
  "translation_languages": {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian"
  },
  "available_translation_pairs": {
    "en": ["es", "fr", "de", "it", "pt", "ru"],
    "es": ["en", "pt", "fr", "de", "it"],
    "fr": ["en", "es", "de", "pt"],
    "de": ["en", "es", "fr", "pt"],
    "it": ["en", "es", "pt"],
    "pt": ["en", "es", "fr", "de", "it"],
    "ru": ["en"]
  }
}
```

## 🏗️ Arquitectura del Proyecto

```
api-transcriptions/
├── app/
│   ├── main.py                 # Punto de entrada de la aplicación
│   ├── api/
│   │   └── endpoints/
│   │       ├── health.py       # Endpoint de salud
│   │       └── transcription.py # Endpoints principales
│   ├── core/
│   │   ├── config.py          # Configuración global
│   │   └── security.py        # Autenticación y seguridad
│   ├── models/
│   │   └── schemas.py         # Modelos Pydantic
│   ├── services/
│   │   ├── transcription_service.py    # Servicio principal Whisper
│   │   ├── subtitle_generator.py       # Generación de SRT
│   │   ├── audio_extractor.py         # Extracción de audio
│   │   ├── quality_evaluator.py       # Evaluación de calidad
│   │   └── translation_service.py     # Servicios de traducción
│   └── utils/
│       ├── file_handler.py     # Manejo de archivos
│       └── validators.py       # Validaciones
├── temp/                       # Archivos temporales
│   ├── uploads/               # Videos subidos
│   ├── audio/                 # Audio extraído
│   ├── srt/                   # Subtítulos generados
│   └── whisper_cache/         # Cache de modelos Whisper
├── tests/                     # Pruebas unitarias
├── requirements.txt           # Dependencias Python
└── README.md                 # Este archivo
```

## 🛠️ Tecnologías Utilizadas

### Core Framework
- **FastAPI** - Framework web moderno y rápido
- **Uvicorn** - Servidor ASGI de alto rendimiento
- **Pydantic** - Validación de datos y serialización

### Procesamiento de Audio/Video
- **OpenAI Whisper** - Reconocimiento de voz de última generación
- **FFmpeg-python** - Manipulación de archivos multimedia
- **PyTorch** - Framework de deep learning

### Traducción
- **Helsinki-NLP/opus-mt** - Modelos de traducción neuronal
- **Transformers (Hugging Face)** - Biblioteca de modelos pre-entrenados
- **MarianMT** - Modelos de traducción automática

### Utilidades
- **aiofiles** - Operaciones de archivos asíncronas
- **python-magic** - Detección de tipos MIME
- **jiwer** - Evaluación de calidad (WER)
- **langdetect** - Detección automática de idiomas

## ⚙️ Configuración

### Variables de Entorno
Crea un archivo `.env` en la raíz del proyecto:

```env
# API Configuration
API_KEY=tu_api_key_aqui
DEBUG=True
HOST=0.0.0.0
PORT=8000

# File Processing
MAX_FILE_SIZE_MB=500
TEMP_DIR=./temp
CLEANUP_TEMP_FILES=True

# Whisper Configuration
WHISPER_MODEL_SIZE=base
WHISPER_DEVICE=auto

# Security
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=3600

# Logging
LOG_LEVEL=INFO
LOG_FILE=transcription_api.log
```

### Modelos Whisper Disponibles
- `tiny` - Más rápido, menor precisión
- `base` - Balance entre velocidad y precisión (recomendado)
- `small` - Buena precisión, velocidad moderada
- `medium` - Alta precisión, más lento
- `large` - Máxima precisión, más lento

## 🧪 Testing

```bash
# Ejecutar todas las pruebas
pytest

# Ejecutar con cobertura
pytest --cov=app

# Ejecutar pruebas específicas
pytest tests/test_transcription.py
```

## 📝 Ejemplos de Uso

### Ejemplo con cURL
```bash
# Transcribir video en inglés y traducir a español
curl -X POST "http://localhost:8000/api/v1/transcribe" \
  -H "X-API-Key: dev_api_key_12345" \
  -F "video_file=@video.mp4" \
  -F "language=auto" \
  -F "translate_to=es" \
  -F "model_size=base"
```

### Ejemplo con Python
```python
import requests

# Subir video para transcripción
response = requests.post(
    "http://localhost:8000/api/v1/transcribe",
    headers={"X-API-Key": "dev_api_key_12345"},
    files={"video_file": open("video.mp4", "rb")},
    data={
        "language": "auto",
        "translate_to": "pt",
        "model_size": "base"
    }
)

job_id = response.json()["job_id"]

# Verificar estado
status_response = requests.get(
    f"http://localhost:8000/api/v1/transcribe/{job_id}/status",
    headers={"X-API-Key": "dev_api_key_12345"}
)

# Descargar subtítulos cuando esté completado
if status_response.json()["status"] == "completed":
    srt_response = requests.get(
        f"http://localhost:8000/api/v1/transcribe/{job_id}/download",
        headers={"X-API-Key": "dev_api_key_12345"}
    )
    
    with open("subtitles.srt", "wb") as f:
        f.write(srt_response.content)
```

## 🔧 Solución de Problemas

### Errores Comunes

#### "FFmpeg not found"
```bash
# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Descargar desde https://ffmpeg.org/download.html
```

#### "Model download failed"
- Verificar conexión a internet
- Los modelos se descargan automáticamente en el primer uso
- Espacio en disco suficiente (modelos grandes pueden ser >1GB)

#### "Translation not working"
- Verificar que el par de idiomas esté soportado en `/api/v1/languages`
- Los modelos de traducción se descargan automáticamente cuando se usan por primera vez

## 📊 Rendimiento

### Tiempos Estimados (video de 1 minuto)
- **tiny**: ~10-15 segundos
- **base**: ~15-25 segundos  
- **small**: ~25-40 segundos
- **medium**: ~40-60 segundos
- **large**: ~60-120 segundos

*Los tiempos varían según el hardware (CPU/GPU) y la complejidad del audio*

## 🤝 Contribuir

1. Fork el proyecto
2. Crear una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abrir un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 🙏 Agradecimientos

- [OpenAI Whisper](https://github.com/openai/whisper) - Reconocimiento de voz
- [Helsinki-NLP](https://huggingface.co/Helsinki-NLP) - Modelos de traducción
- [FastAPI](https://fastapi.tiangolo.com/) - Framework web
- [Hugging Face](https://huggingface.co/) - Biblioteca de transformers

## 📞 Soporte

Si tienes preguntas o problemas:
- Abrir un [Issue](https://github.com/jpcuencat/api-transcriptions/issues)
- Revisar la documentación en `/docs` (Swagger UI)
- Verificar logs en `transcription_api.log`

---

⭐ Si este proyecto te fue útil, ¡dale una estrella en GitHub!
