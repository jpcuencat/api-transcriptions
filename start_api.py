#!/usr/bin/env python3
"""
Script para iniciar la API de transcripción con manejo de errores mejorado
"""
import sys
import os
import signal
import logging
import uvicorn
from pathlib import Path

def setup_graceful_shutdown():
    """Configura manejo de señales para cierre graceful"""
    def signal_handler(signum, frame):
        logging.info(f"Recibida señal {signum}, cerrando aplicación...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def check_dependencies():
    """Verifica que todas las dependencias estén disponibles"""
    try:
        import whisper
        import ffmpeg
        import magic
        logging.info("✅ Todas las dependencias están disponibles")
        return True
    except ImportError as e:
        logging.error(f"❌ Dependencia faltante: {e}")
        return False

def check_system_resources():
    """Verifica recursos del sistema"""
    import psutil
    
    # Verificar memoria
    memory = psutil.virtual_memory()
    available_gb = memory.available / (1024**3)
    
    if available_gb < 2:
        logging.warning(f"⚠️ Poca memoria disponible: {available_gb:.1f}GB")
    else:
        logging.info(f"✅ Memoria disponible: {available_gb:.1f}GB")
    
    # Verificar espacio en disco
    disk = psutil.disk_usage('.')
    free_gb = disk.free / (1024**3)
    
    if free_gb < 5:
        logging.warning(f"⚠️ Poco espacio en disco: {free_gb:.1f}GB")
    else:
        logging.info(f"✅ Espacio en disco: {free_gb:.1f}GB")

def create_directories():
    """Crea directorios necesarios"""
    try:
        directories = [
            "./temp",
            "./temp/uploads", 
            "./temp/audio",
            "./temp/srt",
            "./temp/whisper_cache"
        ]
        
        for dir_path in directories:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            
        logging.info("✅ Directorios creados correctamente")
        return True
    except Exception as e:
        logging.error(f"❌ Error creando directorios: {e}")
        return False

def main():
    """Función principal para iniciar la API"""
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('startup.log')
        ]
    )
    
    logging.info("🚀 Iniciando API de Transcripción de Videos...")
    
    # Verificaciones previas
    if not check_dependencies():
        logging.error("❌ Faltan dependencias críticas. Instalando...")
        os.system("pip install -r requirements.txt")
    
    check_system_resources()
    
    if not create_directories():
        logging.error("❌ No se pudieron crear directorios necesarios")
        sys.exit(1)
    
    # Configurar manejo de señales
    setup_graceful_shutdown()
    
    try:
        logging.info("🌐 Iniciando servidor Uvicorn...")
        
        # Configuración del servidor
        config = uvicorn.Config(
            app="app.main:app",
            host="0.0.0.0", 
            port=8000,
            reload=True,
            reload_dirs=["./app"],
            log_level="info",
            access_log=True,
            use_colors=True,
            timeout_keep_alive=30,
            timeout_notify=30,
            limit_concurrency=10,
            limit_max_requests=1000
        )
        
        server = uvicorn.Server(config)
        
        logging.info("✅ Servidor configurado correctamente")
        logging.info("📡 API disponible en: http://localhost:8000")
        logging.info("📚 Documentación en: http://localhost:8000/docs")
        logging.info("🏥 Health check en: http://localhost:8000/api/v1/health")
        logging.info("🌍 Idiomas soportados en: http://localhost:8000/api/v1/languages")
        logging.info("🔑 API Key para pruebas: dev_api_key_12345")
        
        # Iniciar servidor
        server.run()
        
    except Exception as e:
        logging.error(f"❌ Error crítico al iniciar servidor: {e}")
        logging.exception("Detalles del error:")
        sys.exit(1)

if __name__ == "__main__":
    main()