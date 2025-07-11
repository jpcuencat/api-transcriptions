#!/usr/bin/env python3
"""
Script de diagnóstico para identificar problemas de la API
"""
import sys
import os
import logging
import subprocess
import psutil
from pathlib import Path

def setup_logging():
    """Configura logging para diagnóstico"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def check_python_version():
    """Verifica versión de Python"""
    version = sys.version_info
    logging.info(f"🐍 Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        logging.error("❌ Se requiere Python 3.8 o superior")
        return False
    
    logging.info("✅ Versión de Python compatible")
    return True

def check_virtual_environment():
    """Verifica si está en entorno virtual"""
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        logging.info("✅ Ejecutándose en entorno virtual")
        logging.info(f"📁 Ruta del entorno: {sys.prefix}")
        return True
    else:
        logging.warning("⚠️ No se detectó entorno virtual")
        return False

def check_dependencies():
    """Verifica dependencias críticas"""
    critical_deps = [
        'fastapi',
        'uvicorn',
        'whisper',
        'ffmpeg-python',
        'python-magic',
        'pydantic',
        'pydantic-settings'
    ]
    
    missing_deps = []
    
    for dep in critical_deps:
        try:
            __import__(dep.replace('-', '_'))
            logging.info(f"✅ {dep}: Instalado")
        except ImportError:
            logging.error(f"❌ {dep}: Faltante")
            missing_deps.append(dep)
    
    if missing_deps:
        logging.error(f"❌ Dependencias faltantes: {', '.join(missing_deps)}")
        return False
    
    logging.info("✅ Todas las dependencias están instaladas")
    return True

def check_system_tools():
    """Verifica herramientas del sistema"""
    tools = ['ffmpeg', 'ffprobe']
    
    for tool in tools:
        try:
            result = subprocess.run([tool, '-version'], 
                                  capture_output=True, 
                                  timeout=5)
            if result.returncode == 0:
                logging.info(f"✅ {tool}: Disponible")
            else:
                logging.error(f"❌ {tool}: Error al ejecutar")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logging.error(f"❌ {tool}: No encontrado en PATH")

def check_ports():
    """Verifica si el puerto 8000 está disponible"""
    try:
        import socket
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', 8000))
        sock.close()
        
        if result == 0:
            logging.warning("⚠️ Puerto 8000 ya está en uso")
            
            # Buscar qué proceso usa el puerto
            for conn in psutil.net_connections():
                if conn.laddr.port == 8000:
                    try:
                        process = psutil.Process(conn.pid)
                        logging.warning(f"📱 Proceso usando puerto 8000: {process.name()} (PID: {conn.pid})")
                    except:
                        logging.warning(f"📱 PID usando puerto 8000: {conn.pid}")
            return False
        else:
            logging.info("✅ Puerto 8000 disponible")
            return True
    except Exception as e:
        logging.error(f"❌ Error verificando puerto: {e}")
        return False

def check_directories():
    """Verifica directorios necesarios"""
    dirs = [
        "./temp",
        "./temp/uploads",
        "./temp/audio", 
        "./temp/srt",
        "./temp/whisper_cache",
        "./app",
        "./app/core",
        "./app/api",
        "./app/services"
    ]
    
    all_exist = True
    
    for dir_path in dirs:
        if Path(dir_path).exists():
            logging.info(f"✅ {dir_path}: Existe")
        else:
            logging.error(f"❌ {dir_path}: No existe")
            all_exist = False
    
    return all_exist

def check_files():
    """Verifica archivos críticos"""
    files = [
        "./app/main.py",
        "./app/core/config.py",
        "./requirements.txt",
        "./.env"
    ]
    
    all_exist = True
    
    for file_path in files:
        if Path(file_path).exists():
            size = Path(file_path).stat().st_size
            logging.info(f"✅ {file_path}: Existe ({size} bytes)")
        else:
            logging.error(f"❌ {file_path}: No existe")
            all_exist = False
    
    return all_exist

def check_memory_usage():
    """Verifica uso de memoria"""
    memory = psutil.virtual_memory()
    
    logging.info(f"💾 Memoria total: {memory.total / (1024**3):.1f}GB")
    logging.info(f"💾 Memoria disponible: {memory.available / (1024**3):.1f}GB")
    logging.info(f"💾 Memoria usada: {memory.percent}%")
    
    if memory.available < 2 * (1024**3):  # Menos de 2GB
        logging.warning("⚠️ Poca memoria disponible (< 2GB)")
        return False
    
    return True

def check_disk_space():
    """Verifica espacio en disco"""
    disk = psutil.disk_usage('.')
    
    logging.info(f"💿 Espacio total: {disk.total / (1024**3):.1f}GB")
    logging.info(f"💿 Espacio libre: {disk.free / (1024**3):.1f}GB")
    logging.info(f"💿 Espacio usado: {(disk.used / disk.total) * 100:.1f}%")
    
    if disk.free < 5 * (1024**3):  # Menos de 5GB
        logging.warning("⚠️ Poco espacio en disco (< 5GB)")
        return False
    
    return True

def check_processes():
    """Verifica procesos relacionados con la API"""
    api_processes = []
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = ' '.join(proc.info['cmdline'] or [])
            if 'uvicorn' in cmdline and 'app.main' in cmdline:
                api_processes.append(proc.info)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    if api_processes:
        logging.warning(f"⚠️ Encontrados {len(api_processes)} procesos de la API ejecutándose:")
        for proc in api_processes:
            logging.warning(f"   📱 PID {proc['pid']}: {proc['name']}")
        return False
    else:
        logging.info("✅ No hay procesos de la API ejecutándose")
        return True

def main():
    """Función principal de diagnóstico"""
    setup_logging()
    
    logging.info("🔍 Iniciando diagnóstico de la API de Transcripción...")
    logging.info("=" * 60)
    
    checks = [
        ("Versión de Python", check_python_version),
        ("Entorno Virtual", check_virtual_environment),
        ("Dependencias", check_dependencies),
        ("Herramientas del Sistema", check_system_tools),
        ("Puerto 8000", check_ports),
        ("Directorios", check_directories),
        ("Archivos", check_files),
        ("Memoria", check_memory_usage),
        ("Espacio en Disco", check_disk_space),
        ("Procesos", check_processes)
    ]
    
    results = {}
    
    for check_name, check_func in checks:
        logging.info(f"\n🔍 Verificando: {check_name}")
        logging.info("-" * 40)
        try:
            results[check_name] = check_func()
        except Exception as e:
            logging.error(f"❌ Error en verificación {check_name}: {e}")
            results[check_name] = False
    
    # Resumen
    logging.info("\n" + "=" * 60)
    logging.info("📊 RESUMEN DEL DIAGNÓSTICO")
    logging.info("=" * 60)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for check_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logging.info(f"{status} {check_name}")
    
    logging.info(f"\n🎯 Resultado: {passed}/{total} verificaciones pasaron")
    
    if passed == total:
        logging.info("🎉 ¡Todo está listo para ejecutar la API!")
        logging.info("💡 Ejecuta: python start_api.py")
    else:
        logging.error("⚠️ Hay problemas que necesitan resolverse")
        
        # Sugerencias específicas
        if not results.get("Puerto 8000", True):
            logging.info("💡 Sugerencia: Mata el proceso que usa el puerto 8000")
        
        if not results.get("Dependencias", True):
            logging.info("💡 Sugerencia: pip install -r requirements.txt")
        
        if not results.get("Directorios", True):
            logging.info("💡 Sugerencia: Los directorios se crearán automáticamente")

if __name__ == "__main__":
    main()