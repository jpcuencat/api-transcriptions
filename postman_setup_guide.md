# 🎯 Configuración Correcta de Postman - API de Transcripción

## 🔑 Autenticación: Dos métodos soportados

### Método 1: Bearer Token (RECOMENDADO)
```
Authorization: Bearer dev_api_key_12345
```

### Método 2: X-API-Key Header
```
X-API-Key: dev_api_key_12345
```

---

## 📋 Configuración paso a paso en Postman

### 🚀 1. Health Check (Verificar que funciona)

**Request:**
- Method: `GET`
- URL: `http://localhost:8000/api/v1/health`

**Authorization:**
- Type: `Bearer Token`
- Token: `dev_api_key_12345`

**Respuesta esperada:**
```json
{
  "status": "degraded",
  "services": {
    "api": "running"
  }
}
```

---

### 🌍 2. Obtener Idiomas Soportados

**Request:**
- Method: `GET`
- URL: `http://localhost:8000/api/v1/languages`

**Authorization:**
- Type: `Bearer Token`
- Token: `dev_api_key_12345`

**Respuesta esperada:**
```json
{
  "supported_languages": {
    "auto": "Auto-detect",
    "es": "Spanish",
    "en": "English"
  },
  "whisper_models": ["tiny", "base", "small", "medium", "large"]
}
```

---

### 🎬 3. Transcribir Video (Endpoint principal)

**Request:**
- Method: `POST`
- URL: `http://localhost:8000/api/v1/transcribe`

**Authorization:**
- Type: `Bearer Token`
- Token: `dev_api_key_12345`

**Body (form-data):**
```
video_file: [SELECCIONAR ARCHIVO VIDEO .mp4, .avi, .mov, .mkv, .webm]
language: auto
model_size: base
quality_evaluation: true
translate_to: (opcional - ej: "en")
```

**Configuración en Postman:**
1. Pestaña "Body" → "form-data"
2. Agregar campos:
   - `video_file` → tipo "File" → seleccionar video
   - `language` → tipo "Text" → valor: `auto`
   - `model_size` → tipo "Text" → valor: `base`
   - `quality_evaluation` → tipo "Text" → valor: `true`

**Respuesta inmediata:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing",
  "created_at": "2024-01-15T10:30:00Z"
}
```

---

### 📊 4. Consultar Estado del Trabajo

**Request:**
- Method: `GET`
- URL: `http://localhost:8000/api/v1/transcribe/{{job_id}}/status`

**Authorization:**
- Type: `Bearer Token`
- Token: `dev_api_key_12345`

**Tip:** Guarda el `job_id` de la respuesta anterior como variable:
```javascript
// En pestaña "Tests" del request de transcripción:
pm.test("Save job_id", function () {
    var jsonData = pm.response.json();
    pm.environment.set("job_id", jsonData.job_id);
});
```

**Respuesta cuando está completo:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "transcription_text": "Hola, este es un ejemplo...",
  "detected_language": "es",
  "quality_report": {
    "overall_score": 0.85,
    "quality_level": "good",
    "recommendations": ["Transcription quality is good"]
  },
  "srt_file_path": "/path/to/file.srt",
  "processing_time": 45.2
}
```

---

### 📥 5. Descargar Archivo SRT

**Request:**
- Method: `GET`
- URL: `http://localhost:8000/api/v1/transcribe/{{job_id}}/download`

**Authorization:**
- Type: `Bearer Token`
- Token: `dev_api_key_12345`

**Resultado:** Descarga automática del archivo `.srt`

---

### 🗑️ 6. Limpiar Trabajo (Opcional)

**Request:**
- Method: `DELETE`
- URL: `http://localhost:8000/api/v1/transcribe/{{job_id}}`

**Authorization:**
- Type: `Bearer Token`
- Token: `dev_api_key_12345`

**Respuesta:**
```json
{
  "message": "Job deleted successfully"
}
```

---

## 🎛️ Variables de Entorno en Postman

### Configurar Variables Globales:
1. Click en ⚙️ → "Manage Environments"
2. Click "Add" → Crear "Transcription API"
3. Agregar variables:

```
BASE_URL: http://localhost:8000
API_TOKEN: dev_api_key_12345
```

### Usar en requests:
- URL: `{{BASE_URL}}/api/v1/transcribe`
- Token: `{{API_TOKEN}}`

---

## 🚨 Solución de Errores Comunes

### Error: "Not authenticated"
**Causa:** Falta configuración de autenticación
**Solución:** 
- Verificar que Authorization esté en "Bearer Token"
- Token debe ser exactamente: `dev_api_key_12345`

### Error: "Invalid API key"
**Causa:** Token incorrecto
**Solución:**
- Verificar que no hay espacios extra
- Token correcto: `dev_api_key_12345`

### Error: "Connection refused"
**Causa:** API no está ejecutándose
**Solución:**
```bash
curl http://localhost:8000/api/v1/health
# Si falla, reiniciar API
python restart_api.py
```

### Error: "File too large"
**Causa:** Video > 500MB
**Solución:**
- Usar video más pequeño
- O comprimir el video

---

## ✅ Lista de Verificación

Antes de probar transcripción:

- [ ] API ejecutándose: `curl http://localhost:8000/api/v1/health`
- [ ] Autenticación configurada: Bearer Token
- [ ] Video válido: .mp4, .avi, .mov, .mkv, .webm
- [ ] Tamaño < 500MB
- [ ] Headers correctos en Postman

---

## 🎉 Flujo Completo de Prueba

1. **Health Check** → Verificar API
2. **Languages** → Ver opciones disponibles  
3. **Transcribe** → Subir video → Obtener job_id
4. **Status** → Monitorear progreso (repetir hasta "completed")
5. **Download** → Descargar archivo SRT
6. **Delete** → Limpiar (opcional)

**¡La API optimizada está lista para transcribir videos!** 🚀