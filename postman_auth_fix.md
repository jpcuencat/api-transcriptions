# 🔐 Solución: Error "Not authenticated" en Postman

## ❌ Problema Identificado:
La API usa **Bearer Token** authentication, no header `X-API-Key`.

## ✅ Solución para Postman:

### Opción 1: Usar Authorization Bearer Token (RECOMENDADO)

1. **En Postman, pestaña "Authorization":**
   - Type: `Bearer Token`
   - Token: `dev_api_key_12345`

2. **O en Headers manuales:**
   ```
   Authorization: Bearer dev_api_key_12345
   ```

### Opción 2: Variables de Entorno en Postman

1. **Configurar variable:**
   - Variable: `API_TOKEN`
   - Value: `dev_api_key_12345`

2. **En Authorization:**
   - Type: `Bearer Token`
   - Token: `{{API_TOKEN}}`

## 🧪 Prueba Rápida:

### Request correcto para transcripción:
```http
POST http://localhost:8000/api/v1/transcribe
Authorization: Bearer dev_api_key_12345
Content-Type: multipart/form-data

Body (form-data):
- video_file: [ARCHIVO_VIDEO]
- language: auto
- model_size: base
- quality_evaluation: true
```

### Headers exactos:
```
Authorization: Bearer dev_api_key_12345
```

## 🔧 Configuración paso a paso en Postman:

### 1. Crear nueva request
- Method: `POST`
- URL: `http://localhost:8000/api/v1/transcribe`

### 2. Configurar Authorization
- Click en pestaña "Authorization"
- Type: seleccionar "Bearer Token"
- Token: escribir `dev_api_key_12345`

### 3. Configurar Body
- Click en pestaña "Body"
- Seleccionar "form-data"
- Agregar fields:
  - `video_file` (File): Seleccionar archivo de video
  - `language` (Text): `auto`
  - `model_size` (Text): `base`
  - `quality_evaluation` (Text): `true`

### 4. Enviar request
- Click "Send"

## ✅ Respuesta esperada:
```json
{
  "job_id": "uuid-del-trabajo",
  "status": "processing",
  "created_at": "2024-01-15T10:30:00Z"
}
```

## 🚨 Si sigue dando error:

### Verificar que la API esté funcionando:
```bash
curl -H "Authorization: Bearer dev_api_key_12345" http://localhost:8000/api/v1/health
```

### Respuesta esperada:
```json
{
  "status": "degraded",
  "services": {
    "api": "running"
  }
}
```

## 🔄 Alternativa: Header X-API-Key

Si prefieres usar `X-API-Key` en lugar de Bearer, puedo modificar la API para soportar ambos métodos.