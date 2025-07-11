# 🚀 Guía de Inicio Rápido - API de Transcripción

## ⚡ Problema Identificado: Carga Inicial de Whisper

La aplicación se "cae" porque Whisper está descargando modelos la primera vez. Esto es normal y puede tardar varios minutos.

## 🔧 Soluciones:

### Opción 1: Iniciar sin reload (más estable)
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --log-level info
```

### Opción 2: Usar el script simplificado
```bash
python run_simple.py
```

### Opción 3: Pre-cargar Whisper manualmente
```bash
python -c "
import whisper
print('🔄 Descargando modelo Whisper base...')
model = whisper.load_model('base')
print('✅ Modelo descargado exitosamente!')
"
```

## 📊 Estados de la Aplicación:

### ✅ Funcionando Correctamente
- Puerto 8000 disponible
- Imports sin errores
- Configuración válida
- Dependencias instaladas

### ⏳ Cargando (Normal)
- Primera descarga de modelos Whisper (1-3 minutos)
- Inicialización de servicios
- Creación de directorios

### ❌ Problemas Potenciales
- Puerto 8000 ocupado
- Falta de memoria (< 2GB)
- Dependencias faltantes

## 🎯 Pasos Recomendados:

### 1. Pre-cargar Whisper (RECOMENDADO)
```bash
python -c "
import whisper
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

logger.info('🔄 Descargando modelo base...')
model = whisper.load_model('base')
logger.info('✅ Listo para usar!')
"
```

### 2. Iniciar con monitoreo
```bash
python -c "
import time
import psutil
import subprocess

print('📊 Monitoreando inicio de la API...')
print('🔄 Memoria disponible:', f'{psutil.virtual_memory().available / (1024**3):.1f}GB')

# Iniciar en background
proc = subprocess.Popen(['python', 'run_simple.py'])
print(f'🚀 Proceso iniciado (PID: {proc.pid})')

# Monitorear por 30 segundos
for i in range(30):
    if proc.poll() is not None:
        print(f'❌ Proceso terminó inesperadamente')
        break
    print(f'⏳ Esperando... {i+1}/30s')
    time.sleep(1)

if proc.poll() is None:
    print('✅ API ejecutándose correctamente!')
    print('🌐 Disponible en: http://localhost:8000')
    print('📚 Docs en: http://localhost:8000/docs')
    print('🛑 Presiona Ctrl+C para detener')
    proc.wait()
"
```

### 3. Verificar estado desde otra terminal
```bash
curl http://localhost:8000/api/v1/health
```

## 🐛 Troubleshooting:

### Si sigue cayéndose:
1. **Verificar logs:**
   ```bash
   tail -f transcription_api.log
   ```

2. **Verificar memoria:**
   ```bash
   python -c "import psutil; print(f'Memoria: {psutil.virtual_memory().available / (1024**3):.1f}GB')"
   ```

3. **Verificar puerto:**
   ```bash
   python -c "
   import socket
   sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
   result = sock.connect_ex(('localhost', 8000))
   print('Puerto 8000:', 'OCUPADO' if result == 0 else 'DISPONIBLE')
   sock.close()
   "
   ```

### Si Whisper falla:
```bash
# Reinstalar Whisper
pip uninstall openai-whisper -y
pip install openai-whisper==20231117

# Limpiar cache
rm -rf ~/.cache/whisper/
rm -rf ./temp/whisper_cache/
```

## ⚡ Inicio Ultra-Rápido para Pruebas:

Si solo quieres probar la API rápidamente:

```bash
# Terminal 1: Pre-cargar Whisper
python -c "import whisper; whisper.load_model('base'); print('✅ Listo!')"

# Terminal 2: Iniciar API
python run_simple.py
```

En 1-2 minutos tendrás la API funcionando estable.

## 🎉 Una vez funcionando:

- ✅ Health: `GET http://localhost:8000/api/v1/health`
- ✅ Docs: `http://localhost:8000/docs`
- ✅ Idiomas: `GET http://localhost:8000/api/v1/languages`
- ✅ Transcribir: `POST http://localhost:8000/api/v1/transcribe`

**API Key para pruebas:** `dev_api_key_12345`