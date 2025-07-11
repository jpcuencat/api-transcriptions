#!/usr/bin/env python3
"""
Script para probar la extracción de audio y identificar el problema
"""
import os
import sys
import logging
import subprocess
import asyncio
import tempfile
from pathlib import Path

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_ffmpeg_direct():
    """Prueba FFmpeg directamente"""
    try:
        logger.info("🔍 Probando FFmpeg directamente...")
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            version = result.stdout.split('\n')[0]
            logger.info(f"✅ FFmpeg: {version}")
            return True
        else:
            logger.error(f"❌ FFmpeg error: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"❌ FFmpeg test failed: {e}")
        return False

def test_video_info(video_path):
    """Prueba obtener información del video"""
    try:
        logger.info(f"📹 Probando información del video: {video_path}")
        
        if not os.path.exists(video_path):
            logger.error(f"❌ Video no encontrado: {video_path}")
            return None
        
        # Usar ffprobe para obtener información
        cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', video_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            import json
            info = json.loads(result.stdout)
            
            # Extraer información relevante
            format_info = info.get('format', {})
            streams = info.get('streams', [])
            
            video_stream = next((s for s in streams if s.get('codec_type') == 'video'), None)
            audio_stream = next((s for s in streams if s.get('codec_type') == 'audio'), None)
            
            video_info = {
                'duration': float(format_info.get('duration', 0)),
                'size': int(format_info.get('size', 0)),
                'video_codec': video_stream.get('codec_name') if video_stream else None,
                'audio_codec': audio_stream.get('codec_name') if audio_stream else None,
                'video_streams': len([s for s in streams if s.get('codec_type') == 'video']),
                'audio_streams': len([s for s in streams if s.get('codec_type') == 'audio'])
            }
            
            logger.info(f"✅ Video info: {video_info}")
            return video_info
        else:
            logger.error(f"❌ ffprobe error: {result.stderr}")
            return None
            
    except Exception as e:
        logger.error(f"❌ Error getting video info: {e}")
        return None

def test_audio_extraction_direct(video_path, output_path):
    """Prueba extracción de audio con subprocess directo"""
    try:
        logger.info(f"🎵 Probando extracción de audio...")
        logger.info(f"📹 Entrada: {video_path}")
        logger.info(f"🔊 Salida: {output_path}")
        
        # Comando FFmpeg optimizado
        cmd = [
            'ffmpeg', '-y',  # Overwrite
            '-i', video_path,
            '-acodec', 'pcm_s16le',  # PCM 16-bit
            '-ar', '16000',          # 16kHz sample rate
            '-ac', '1',              # Mono
            '-t', '30',              # Limitar a 30 segundos para prueba
            '-loglevel', 'info',     # Verbose para debug
            output_path
        ]
        
        logger.info(f"🔧 Comando: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        logger.info(f"📤 Return code: {result.returncode}")
        if result.stdout:
            logger.info(f"📤 Stdout: {result.stdout}")
        if result.stderr:
            logger.info(f"📤 Stderr: {result.stderr}")
        
        if result.returncode == 0:
            if os.path.exists(output_path):
                size = os.path.getsize(output_path)
                logger.info(f"✅ Audio extraído exitosamente: {size} bytes")
                return True
            else:
                logger.error("❌ FFmpeg exitoso pero archivo no creado")
                return False
        else:
            logger.error(f"❌ FFmpeg falló con código {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("❌ Timeout en extracción de audio")
        return False
    except Exception as e:
        logger.error(f"❌ Error en extracción: {e}")
        return False

async def test_audio_extractor_service(video_path):
    """Prueba el AudioExtractor de la API"""
    try:
        logger.info("🔧 Probando AudioExtractor de la API...")
        
        sys.path.append('/mnt/d/api-transcriptions')
        from app.services.audio_extractor import AudioExtractor
        
        extractor = AudioExtractor()
        
        # Crear archivo temporal para salida
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            output_path = tmp.name
        
        try:
            result = await extractor.extract_audio(video_path, output_path)
            
            if os.path.exists(output_path):
                size = os.path.getsize(output_path)
                logger.info(f"✅ AudioExtractor exitoso: {size} bytes")
                return True
            else:
                logger.error("❌ AudioExtractor: archivo no creado")
                return False
                
        finally:
            # Limpiar archivo temporal
            if os.path.exists(output_path):
                os.unlink(output_path)
        
    except Exception as e:
        logger.error(f"❌ AudioExtractor falló: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Función principal de diagnóstico"""
    logger.info("🔍 Diagnóstico de Extracción de Audio")
    logger.info("=" * 50)
    
    # Buscar el video que está fallando
    video_patterns = [
        "/mnt/d/api-transcriptions/media/AC20100LabIntro1080p.mp4",
        "/mnt/d/api-transcriptions/media/AC201 12 Data Loading (1080p).mp4",
        "./temp/uploads/*AC20100LabIntro1080p.mp4"
    ]
    
    video_path = None
    for pattern in video_patterns:
        if '*' in pattern:
            import glob
            matches = glob.glob(pattern)
            if matches:
                video_path = matches[0]
                break
        elif os.path.exists(pattern):
            video_path = pattern
            break
    
    if not video_path:
        logger.error("❌ No se encontró el video de prueba")
        logger.info("💡 Creando video de prueba sintético...")
        
        # Crear video de prueba
        test_video = "/tmp/test_video.mp4"
        cmd = [
            'ffmpeg', '-y',
            '-f', 'lavfi',
            '-i', 'testsrc2=duration=10:size=320x240:rate=30',
            '-f', 'lavfi',
            '-i', 'sine=frequency=440:duration=10',
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-shortest',
            test_video
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0 and os.path.exists(test_video):
                logger.info(f"✅ Video de prueba creado: {test_video}")
                video_path = test_video
            else:
                logger.error("❌ No se pudo crear video de prueba")
                return
        except Exception as e:
            logger.error(f"❌ Error creando video de prueba: {e}")
            return
    
    logger.info(f"🎬 Usando video: {video_path}")
    
    # 1. Test FFmpeg
    if not test_ffmpeg_direct():
        logger.error("❌ FFmpeg no funciona, no se puede continuar")
        return
    
    # 2. Test video info
    video_info = test_video_info(video_path)
    if not video_info:
        logger.error("❌ No se puede obtener info del video")
        return
    
    # 3. Test audio extraction direct
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        output_path = tmp.name
    
    try:
        if test_audio_extraction_direct(video_path, output_path):
            logger.info("✅ Extracción directa exitosa")
        else:
            logger.error("❌ Extracción directa falló")
            return
    finally:
        if os.path.exists(output_path):
            os.unlink(output_path)
    
    # 4. Test AudioExtractor service
    async def run_service_test():
        return await test_audio_extractor_service(video_path)
    
    if asyncio.run(run_service_test()):
        logger.info("✅ AudioExtractor service funcionando")
        logger.info("🎉 ¡Todos los tests de audio pasaron!")
        logger.info("💡 El problema del 'Broken pipe' puede estar en Whisper")
    else:
        logger.error("❌ AudioExtractor service falló")
    
    # Limpiar video de prueba temporal
    if video_path.startswith('/tmp/'):
        if os.path.exists(video_path):
            os.unlink(video_path)

if __name__ == "__main__":
    main()