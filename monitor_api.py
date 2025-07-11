#!/usr/bin/env python3
"""
Script para monitorear y reiniciar automáticamente la API
"""
import time
import socket
import subprocess
import logging
import signal
import sys
import psutil
from datetime import datetime

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('api_monitor.log')
    ]
)
logger = logging.getLogger(__name__)

class APIMonitor:
    def __init__(self):
        self.api_process = None
        self.restart_count = 0
        self.max_restarts = 5
        self.check_interval = 10  # segundos
        self.startup_timeout = 60  # segundos para considerarse "iniciado"
        
    def is_port_active(self, port=8000):
        """Verifica si el puerto está activo"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex(('localhost', port))
            sock.close()
            return result == 0
        except Exception:
            return False
    
    def kill_existing_processes(self):
        """Mata procesos existentes de la API"""
        killed = 0
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = ' '.join(proc.info['cmdline'] or [])
                if ('uvicorn' in cmdline and 'app.main' in cmdline) or 'run_simple.py' in cmdline:
                    logger.info(f"🔪 Matando proceso PID {proc.info['pid']}: {proc.info['name']}")
                    psutil.Process(proc.info['pid']).terminate()
                    killed += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        if killed > 0:
            time.sleep(3)  # Esperar a que terminen
            logger.info(f"🧹 {killed} procesos terminados")
        
        return killed
    
    def start_api(self):
        """Inicia la API"""
        try:
            # Limpiar procesos previos
            self.kill_existing_processes()
            
            logger.info("🚀 Iniciando API...")
            
            # Iniciar proceso
            self.api_process = subprocess.Popen(
                ['python', 'run_simple.py'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Esperar que inicie
            start_time = time.time()
            while time.time() - start_time < self.startup_timeout:
                if self.api_process.poll() is not None:
                    # Proceso terminó prematuramente
                    stdout, stderr = self.api_process.communicate()
                    logger.error(f"❌ API terminó durante el inicio:")
                    logger.error(f"STDOUT: {stdout}")
                    logger.error(f"STDERR: {stderr}")
                    return False
                
                if self.is_port_active():
                    logger.info("✅ API iniciada correctamente")
                    self.restart_count = 0  # Reset contador en éxito
                    return True
                
                time.sleep(2)
            
            logger.error("❌ Timeout esperando que la API inicie")
            return False
            
        except Exception as e:
            logger.error(f"❌ Error iniciando API: {e}")
            return False
    
    def check_api_health(self):
        """Verifica si la API está funcionando"""
        try:
            # Verificar proceso
            if self.api_process and self.api_process.poll() is not None:
                logger.warning("⚠️ Proceso de API terminó")
                return False
            
            # Verificar puerto
            if not self.is_port_active():
                logger.warning("⚠️ Puerto 8000 no responde")
                return False
            
            # Verificar endpoint health
            try:
                import requests
                response = requests.get('http://localhost:8000/api/v1/health', timeout=10)
                if response.status_code == 200:
                    return True
                else:
                    logger.warning(f"⚠️ Health check failed: {response.status_code}")
                    return False
            except Exception:
                logger.warning("⚠️ Health endpoint no responde")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error verificando health: {e}")
            return False
    
    def monitor(self):
        """Loop principal de monitoreo"""
        logger.info("🔍 Iniciando monitor de API...")
        
        # Configurar manejo de señales
        def signal_handler(signum, frame):
            logger.info("🛑 Deteniendo monitor...")
            if self.api_process:
                self.api_process.terminate()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Loop de monitoreo
        while True:
            try:
                if not self.check_api_health():
                    logger.warning(f"💔 API no está funcionando (reinicio #{self.restart_count + 1})")
                    
                    if self.restart_count >= self.max_restarts:
                        logger.error(f"❌ Máximo de reinicios alcanzado ({self.max_restarts})")
                        logger.error("🚨 La API tiene problemas serios, revisa los logs")
                        break
                    
                    self.restart_count += 1
                    
                    if self.start_api():
                        logger.info(f"🎉 API reiniciada exitosamente (#{self.restart_count})")
                    else:
                        logger.error(f"❌ Fallo al reiniciar API (#{self.restart_count})")
                        time.sleep(30)  # Esperar más tiempo antes del siguiente intento
                else:
                    # API funcionando bien
                    logger.info("💚 API funcionando correctamente")
                
                time.sleep(self.check_interval)
                
            except KeyboardInterrupt:
                logger.info("🛑 Monitor detenido por usuario")
                break
            except Exception as e:
                logger.error(f"❌ Error en loop de monitoreo: {e}")
                time.sleep(self.check_interval)
        
        # Cleanup
        if self.api_process:
            self.api_process.terminate()
        logger.info("👋 Monitor finalizado")

def main():
    """Función principal"""
    logger.info("🔧 Iniciando Monitor de API de Transcripción")
    logger.info("=" * 50)
    
    # Mostrar información del sistema
    memory = psutil.virtual_memory()
    logger.info(f"💾 Memoria disponible: {memory.available / (1024**3):.1f}GB")
    
    disk = psutil.disk_usage('.')
    logger.info(f"💿 Espacio libre: {disk.free / (1024**3):.1f}GB")
    
    logger.info("📡 URL de la API: http://localhost:8000")
    logger.info("📚 Documentación: http://localhost:8000/docs")
    logger.info("🔑 API Key: dev_api_key_12345")
    logger.info("=" * 50)
    
    monitor = APIMonitor()
    monitor.monitor()

if __name__ == "__main__":
    main()