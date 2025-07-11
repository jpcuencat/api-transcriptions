#!/usr/bin/env python3
"""
Script simple para iniciar la API sin configuraciones complejas
"""
import uvicorn
import logging

# Configurar logging simple
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("🚀 Iniciando API de transcripción...")
    logger.info("📡 Disponible en: http://localhost:8000")
    logger.info("📚 Docs en: http://localhost:8000/docs")
    
    try:
        uvicorn.run(
            "app.main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        logger.info("🛑 Aplicación detenida por el usuario")
    except Exception as e:
        logger.error(f"❌ Error: {e}")