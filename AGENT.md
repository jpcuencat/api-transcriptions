# 🤖 Contexto Técnico del Proyecto API Transcriptions

## 📋 Descripción General
API desarrollada en Python con **FastAPI** para transcripción automática de videos usando **Whisper** de OpenAI, traducción multiidioma con modelos **Helsinki-NLP**, y generación de subtítulos SRT con tiempos precisos de Whisper.

## 🏗️ Arquitectura Técnica

### Componentes Principales

#### 🎯 **Core API (FastAPI)**
- **Endpoints asíncronos** con procesamiento en background
- **Autenticación por API Key** con rate limiting
- **Validación robusta** con Pydantic
- **Manejo de errores** estructurado
- **Documentación automática** con Swagger/OpenAPI

#### 🎵 **Procesamiento de Audio/Video**
- **FFmpeg-python**: Extracción de audio de videos
- **Whisper**: Transcripción de alta calidad con confidence scores
- **Detección automática** de idiomas
- **Soporte múltiples formatos**: mp4, avi, mov, mkv, webm

#### 🌍 **Sistema de Traducción**
- **Modelos locales Helsinki-NLP** (opus-mt-*) con descarga automática
- **Mapeo completo** de pares de idiomas soportados
- **Sistema de fallback** con diccionarios básicos
- **Cache local** de modelos para rendimiento

#### 📝 **Generación de Subtítulos**
- **Tiempos precisos** usando segmentos originales de Whisper
- **Optimización inteligente** de texto para legibilidad
- **Formato SRT estándar** compatible
- **Preservación completa** del contenido (sin pérdida de segmentos)

## 🔧 Implementación Técnica

### Flujo de Procesamiento
1. **Upload & Validación**: Archivo recibido, validado (tipo, tamaño, formato)
2. **Extracción Audio**: FFmpeg extrae audio WAV del video
3. **Transcripción**: Whisper procesa audio → texto + segmentos con tiempos
4. **Traducción** (opcional): Helsinki-NLP traduce texto + segmentos individuales
5. **Generación SRT**: Subtítulos con tiempos exactos de Whisper
6. **Evaluación Calidad**: WER score y métricas de confianza
7. **Limpieza**: Archivos temporales eliminados automáticamente

### Estructura de Datos

#### Segmentos de Whisper
```python
{
    "id": 0,
    "start": 0.0,
    "end": 6.46,
    "text": "Data loading. Agenda.",
    "confidence": 0.8,
    "translation": "Carga de datos. Agenda."  # Si aplica
}
```

#### Respuesta de Transcripción
```python
{
    "job_id": "uuid",
    "status": "completed",
    "transcription_text": "Texto completo...",
    "translation_text": "Texto traducido...",
    "detected_language": "en",
    "segments": [...],  # Segmentos originales
    "translation_segments": [...],  # Segmentos traducidos
    "quality_report": {...},
    "srt_file_path": "path/to/file.srt"
}
```

## 🛠️ Stack Tecnológico Detallado

### Framework y Core
- **FastAPI 0.104+**: API moderna con type hints
- **Uvicorn**: Servidor ASGI de alto rendimiento
- **Pydantic 2.0+**: Validación y serialización de datos
- **Python 3.8+**: Soporte para async/await nativo

### Procesamiento ML/AI
- **openai-whisper**: Modelo de transcripción de última generación
- **torch**: PyTorch para inferencia de modelos
- **transformers**: Biblioteca Hugging Face para modelos NLP
- **MarianMTModel**: Modelos de traducción neuronal
- **Helsinki-NLP/opus-mt**: Modelos específicos de traducción

### Multimedia y Archivos
- **ffmpeg-python**: Wrapper Python para FFmpeg
- **aiofiles**: Operaciones de archivos asíncronas
- **python-magic**: Detección robusta de tipos MIME
- **moviepy**: Procesamiento adicional de video (backup)

### Calidad y Evaluación
- **jiwer**: Cálculo de Word Error Rate (WER)
- **langdetect**: Detección automática de idiomas
- **numpy**: Operaciones numéricas para métricas

### Utilidades y Monitoreo
- **psutil**: Monitoreo de recursos del sistema
- **rich**: Logging colorizado y mejorado
- **httpx**: Cliente HTTP asíncrono
- **python-multipart**: Manejo de form-data

## ⚙️ Configuración Técnica

### Idiomas Soportados (Configurados)
```python
SUPPORTED_LANGUAGES = {
    "auto": "Auto-detect",
    "es": "Spanish", "en": "English", "fr": "French",
    "de": "German", "it": "Italian", "pt": "Portuguese",
    "zh": "Chinese", "ja": "Japanese", "ko": "Korean",
    "ru": "Russian", "ar": "Arabic"
}
```

### Modelos de Traducción Mapeados
```python
# Pares disponibles con modelos Helsinki-NLP
EN ↔ [ES, FR, DE, IT, PT, RU]
ES ↔ [EN, PT, FR, DE, IT]
FR ↔ [EN, ES, DE, PT]
DE ↔ [EN, ES, FR, PT]
IT ↔ [EN, ES, PT]
PT ↔ [EN, ES, FR, DE, IT]
RU ↔ [EN]
```

### Configuración de Rendimiento
```python
# Whisper
WHISPER_DEVICE = "auto"  # auto/cpu/cuda
WHISPER_CACHE_DIR = "./temp/whisper_cache"

# Archivos
MAX_FILE_SIZE_MB = 500
MAX_CONCURRENT_JOBS = 3
JOB_TIMEOUT_SECONDS = 3600

# Rate Limiting
RATE_LIMIT_REQUESTS = 100
RATE_LIMIT_WINDOW = 3600  # 1 hora
```

## 🔄 Patrones de Diseño Implementados

### 1. **Service Layer Pattern**
- Separación clara entre endpoints y lógica de negocio
- Servicios especializados e independientes
- Inyección de dependencias implícita

### 2. **Repository Pattern** (Archivo)
- `FileHandler`: Abstracción para operaciones de archivos
- Rutas generadas automáticamente y consistentes
- Limpieza automática de archivos temporales

### 3. **Strategy Pattern** (Traducción)
- Múltiples estrategias de traducción
- Fallback automático entre métodos
- Extensible para nuevos proveedores

### 4. **Factory Pattern** (Modelos)
- Carga dinámica de modelos Whisper
- Cache inteligente de modelos pesados
- Configuración flexible de dispositivos

## 🚀 Optimizaciones Implementadas

### Rendimiento
- **Cache de modelos**: Whisper + Helsinki-NLP cargados una vez
- **Procesamiento asíncrono**: Background tasks no bloqueantes
- **Streaming de archivos**: Upload/download eficiente
- **Limpieza automática**: Gestión de memoria y almacenamiento

### Calidad
- **Preservación de segmentos**: Sin pérdida de contenido en SRT
- **Tiempos precisos**: Uso directo de timestamps de Whisper
- **Validación robusta**: Archivos, formatos, parámetros
- **Manejo de errores**: Recuperación elegante de fallos

### Escalabilidad
- **Rate limiting**: Protección contra abuso
- **Job storage**: Sistema de trabajos en memoria (extensible a DB)
- **Resource monitoring**: Tracking de uso de CPU/memoria
- **Logging estructurado**: Trazabilidad completa

## 🧪 Testing y Calidad

### Estructura de Tests
```
tests/
├── conftest.py          # Configuración compartida
├── test_unit.py         # Tests unitarios
├── test_integration.py  # Tests de integración
└── test_transcription.py # Tests específicos de transcripción
```

### Métricas de Calidad
- **Cobertura de código**: Target >80%
- **Word Error Rate**: Evaluación automática de precisión
- **Confidence scores**: Métricas de confianza por segmento
- **Response times**: Benchmarking de rendimiento

## 🔒 Seguridad Implementada

### Autenticación
- API Key validation en todos los endpoints
- Rate limiting por IP/usuario
- Sanitización de inputs

### Archivos
- Validación estricta de tipos MIME
- Límites de tamaño de archivo
- Limpieza automática de temporales
- Paths seguros y controlados

### Datos
- No persistencia de datos sensibles
- Logs sin información personal
- Configuración externalizada

## 📊 Monitoreo y Observabilidad

### Logging
- Logs estructurados con niveles
- Rotación automática de archivos
- Trazabilidad completa de requests
- Performance metrics por operación

### Métricas Disponibles
- Tiempo de procesamiento por modelo
- Tasa de éxito/error por endpoint
- Uso de recursos (CPU/memoria/disco)
- Distribución de idiomas procesados

## 🔮 Extensibilidad

### Puntos de Extensión
1. **Nuevos Modelos**: Agregar modelos Whisper custom
2. **Proveedores Traducción**: Integrar APIs externas
3. **Formatos Output**: Soporte VTT, ASS, etc.
4. **Storage**: Migrar a S3, PostgreSQL, Redis
5. **Notificaciones**: Webhooks, email, etc.

### Arquitectura Preparada Para
- Microservicios (separación por dominio)
- Containerización (Docker/Kubernetes)
- Caching distribuido (Redis)
- Message queues (RabbitMQ/Kafka)
- Bases de datos (PostgreSQL/MongoDB)

---

Este documento proporciona el contexto técnico completo para desarrolladores que trabajen en el proyecto, asegurando comprensión profunda de la arquitectura, decisiones de diseño y posibilidades de extensión.
