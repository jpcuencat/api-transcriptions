#!/usr/bin/env python3
"""
Script para probar y diagnosticar el servicio de transcripción
"""
import os
import sys
import logging
import tempfile
import subprocess
import torch
import whisper
from pathlib import Path

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_audio():
    """Crea un archivo de audio de prueba"""
    try:
        logger.info("🎵 Creando archivo de audio de prueba...")
        
        # Crear un archivo WAV de prueba de 5 segundos con ffmpeg
        test_audio_path = "/tmp/test_audio.wav"
        
        cmd = [
            'ffmpeg', '-y',  # Sobrescribir si existe
            '-f', 'lavfi',   # Usar generador interno
            '-i', 'sine=frequency=440:duration=5',  # Tono de 440Hz por 5 segundos
            '-ar', '16000',  # Sample rate 16kHz (lo que Whisper espera)
            '-ac', '1',      # Mono
            test_audio_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Error creando audio de prueba: {result.stderr}")
            return None
        
        if os.path.exists(test_audio_path):
            size = os.path.getsize(test_audio_path)
            logger.info(f"✅ Audio de prueba creado: {test_audio_path} ({size} bytes)")
            return test_audio_path
        else:
            logger.error("❌ El archivo de audio no se creó")
            return None
            
    except Exception as e:
        logger.error(f"❌ Error creando audio de prueba: {e}")
        return None

def test_whisper_model():
    """Prueba la carga del modelo Whisper"""
    try:
        logger.info("🤖 Probando carga del modelo Whisper...")
        
        # Cargar modelo con configuración segura
        model = whisper.load_model("base", device="cpu")
        logger.info("✅ Modelo Whisper cargado exitosamente")
        
        return model
        
    except Exception as e:
        logger.error(f"❌ Error cargando modelo Whisper: {e}")
        return None

def test_transcription_with_real_audio(audio_path):
    """Prueba transcripción con archivo de audio real"""
    try:
        logger.info(f"🎤 Probando transcripción con: {audio_path}")
        
        if not os.path.exists(audio_path):
            logger.error(f"❌ Archivo no encontrado: {audio_path}")
            return None
        
        size = os.path.getsize(audio_path)
        logger.info(f"📁 Tamaño del archivo: {size} bytes")
        
        # Cargar modelo
        model = test_whisper_model()
        if not model:
            return None
        
        # Configuración segura para transcripción
        options = {
            'fp16': False,
            'verbose': True,
            'word_timestamps': False,
            'temperature': 0.0,
            'language': None  # Auto-detectar
        }
        
        logger.info("🔄 Iniciando transcripción...")
        result = model.transcribe(audio_path, **options)
        
        logger.info("✅ Transcripción completada!")
        logger.info(f"📝 Texto: {result.get('text', 'No text')}")
        logger.info(f"🌍 Idioma detectado: {result.get('language', 'Unknown')}")
        logger.info(f"📊 Segmentos: {len(result.get('segments', []))}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Error en transcripción: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_transcription_service():
    """Prueba el servicio de transcripción de la API"""
    try:
        logger.info("🔧 Probando TranscriptionService de la API...")
        
        sys.path.append('/mnt/d/api-transcriptions')
        from app.services.transcription_service import TranscriptionService
        
        service = TranscriptionService()
        
        # Crear audio de prueba
        test_audio_path = create_test_audio()
        if not test_audio_path:
            logger.error("❌ No se pudo crear audio de prueba")
            return False
        
        # Probar transcripción con el servicio
        import asyncio
        
        async def run_test():
            result = await service.transcribe_audio(
                test_audio_path,
                language='auto',
                model_size='base'
            )
            return result
        
        result = asyncio.run(run_test())
        
        logger.info("✅ TranscriptionService funcionando!")
        logger.info(f"📝 Resultado: {result}")
        
        # Limpiar archivo de prueba
        if os.path.exists(test_audio_path):
            os.remove(test_audio_path)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error en TranscriptionService: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Función principal de diagnóstico"""
    logger.info("🔍 Diagnóstico del Sistema de Transcripción")
    logger.info("=" * 50)
    
    # 1. Verificar PyTorch
    logger.info(f"🔥 PyTorch version: {torch.__version__}")
    logger.info(f"🔥 CUDA disponible: {torch.cuda.is_available()}")
    
    # 2. Verificar Whisper
    logger.info(f"🤖 Whisper version: {whisper.__version__}")
    
    # 3. Verificar FFmpeg
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            logger.info(f"🎬 FFmpeg: {version_line}")
        else:
            logger.error("❌ FFmpeg no disponible")
    except:
        logger.error("❌ FFmpeg no encontrado")
    
    # 4. Probar carga de modelo
    model = test_whisper_model()
    if not model:
        logger.error("❌ No se puede continuar sin modelo Whisper")
        return
    
    # 5. Crear y probar con audio sintético
    test_audio_path = create_test_audio()
    if test_audio_path:
        test_transcription_with_real_audio(test_audio_path)
        # Limpiar
        if os.path.exists(test_audio_path):
            os.remove(test_audio_path)
    
    # 6. Probar servicio de la API
    logger.info("\n" + "=" * 50)
    logger.info("🔧 Probando servicio de la API...")
    if test_transcription_service():
        logger.info("🎉 ¡Todos los tests pasaron!")
        logger.info("💡 El problema puede estar en la configuración específica del archivo")
    else:
        logger.error("❌ Falló el test del servicio")
        logger.info("💡 Revisa los logs para más detalles")

if __name__ == "__main__":
    main()