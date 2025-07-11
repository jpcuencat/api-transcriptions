#!/usr/bin/env python3
"""
Script para reiniciar la API rápidamente
"""
import os
import sys
import time
import socket
import subprocess
import psutil
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_port(port=8000):
    """Verifica si el puerto está activo"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', port))
        sock.close()
        return result == 0
    except:
        return False

def kill_api_processes():
    """Mata todos los procesos de la API"""
    killed = 0
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = ' '.join(proc.info['cmdline'] or [])
            if any(keyword in cmdline for keyword in ['uvicorn', 'app.main', 'run_simple.py', 'start_api.py']):
                logger.info(f"🔪 Matando PID {proc.info['pid']}: {proc.info['name']}")
                psutil.Process(proc.info['pid']).terminate()
                killed += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    if killed > 0:
        time.sleep(3)
        logger.info(f"🧹 {killed} procesos terminados")
    
    return killed

def start_api():
    """Inicia la API"""
    logger.info("🚀 Iniciando API...")
    
    # Cambiar al directorio correcto
    os.chdir('/mnt/d/api-transcriptions')
    
    # Iniciar proceso en background
    process = subprocess.Popen(
        ['python', 'run_simple.py'],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    
    # Esperar un momento y verificar
    for i in range(20):  # 20 segundos máximo
        time.sleep(1)
        if check_port():
            logger.info("✅ API iniciada correctamente!")
            logger.info("📡 Disponible en: http://localhost:8000")
            logger.info("📚 Docs en: http://localhost:8000/docs")
            return True
        
        if process.poll() is not None:
            logger.error("❌ Proceso terminó prematuramente")
            return False
        
        print(f"⏳ Esperando... {i+1}/20")
    
    logger.error("❌ Timeout esperando que inicie")
    return False

def main():
    """Función principal"""
    logger.info("🔧 Reiniciador de API de Transcripción")
    logger.info("=" * 40)
    
    # Verificar estado actual
    if check_port():
        logger.info("ℹ️ API ya está funcionando en puerto 8000")
        response = input("¿Quieres reiniciarla de todos modos? (y/N): ")
        if response.lower() != 'y':
            logger.info("👋 Operación cancelada")
            return
    
    # Limpiar procesos existentes
    logger.info("🧹 Limpiando procesos existentes...")
    killed = kill_api_processes()
    
    # Verificar que el puerto esté libre
    if check_port():
        logger.error("❌ Puerto 8000 sigue ocupado después de limpiar procesos")
        logger.error("💡 Tip: Usa 'sudo lsof -ti:8000 | xargs kill -9' si es necesario")
        return
    
    # Iniciar API
    if start_api():
        logger.info("🎉 ¡API reiniciada exitosamente!")
        
        # Probar endpoint
        try:
            import requests
            response = requests.get('http://localhost:8000/api/v1/health', timeout=5)
            logger.info(f"✅ Health check: {response.status_code}")
        except Exception as e:
            logger.warning(f"⚠️ Error en health check: {e}")
            logger.info("💡 La API puede estar iniciando todavía, espera unos segundos")
    else:
        logger.error("❌ Fallo al reiniciar la API")
        logger.info("🔍 Revisa los logs con: tail -f transcription_api.log")

if __name__ == "__main__":
    main()