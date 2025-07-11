import ffmpeg
import os
import logging
from typing import Dict
from app.core.config import settings

class AudioExtractor:
    def __init__(self):
        self.supported_formats = settings.ALLOWED_EXTENSIONS
        
    async def extract_audio(self, 
                          video_path: str, 
                          output_path: str,
                          sample_rate: int = 16000) -> str:
        """Extrae audio de un video usando FFmpeg con múltiples métodos"""
        try:
            logging.info(f"Extracting audio from {video_path}")
            
            # Verificar que el archivo de entrada existe
            if not os.path.exists(video_path):
                raise Exception(f"Video file not found: {video_path}")
            
            # Método 1: Usar subprocess directamente (más robusto)
            try:
                import subprocess
                cmd = [
                    'ffmpeg', '-y',  # Overwrite output
                    '-i', video_path,
                    '-acodec', 'pcm_s16le',  # PCM 16-bit
                    '-ar', str(sample_rate),  # Sample rate
                    '-ac', '1',              # Mono
                    '-loglevel', 'error',    # Reduce logging
                    output_path
                ]
                
                logging.info(f"Running FFmpeg command: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                        logging.info(f"Audio extracted successfully: {output_path}")
                        return output_path
                    else:
                        raise Exception("FFmpeg completed but output file is empty or missing")
                else:
                    logging.error(f"FFmpeg stderr: {result.stderr}")
                    raise Exception(f"FFmpeg failed with code {result.returncode}: {result.stderr}")
                    
            except subprocess.TimeoutExpired:
                raise Exception("FFmpeg timeout - video too long or processing too slow")
            except Exception as subprocess_error:
                logging.warning(f"Subprocess method failed: {subprocess_error}")
                
                # Método 2: Fallback a ffmpeg-python
                logging.info("Trying fallback method with ffmpeg-python...")
                stream = ffmpeg.input(video_path)
                stream = ffmpeg.output(
                    stream, 
                    output_path,
                    acodec='pcm_s16le',
                    ar=sample_rate,
                    ac=1,
                    loglevel='error'
                )
                
                ffmpeg.run(stream, overwrite_output=True, quiet=True)
                
                if not os.path.exists(output_path):
                    raise Exception("Both extraction methods failed")
                
                logging.info(f"Audio extracted with fallback method: {output_path}")
                return output_path
            
        except ffmpeg.Error as e:
            logging.error(f"FFmpeg error: {e}")
            raise Exception(f"Error extracting audio: {e}")
        except Exception as e:
            logging.error(f"Audio extraction error: {e}")
            raise Exception(f"Audio extraction failed: {e}")
    
    def get_video_info(self, video_path: str) -> Dict:
        """Obtiene información del video"""
        try:
            probe = ffmpeg.probe(video_path)
            
            video_info = next(
                (stream for stream in probe['streams'] 
                 if stream['codec_type'] == 'video'), None
            )
            audio_info = next(
                (stream for stream in probe['streams'] 
                 if stream['codec_type'] == 'audio'), None
            )
            
            duration = float(probe['format']['duration'])
            size = int(probe['format']['size'])
            
            return {
                'duration': duration,
                'size': size,
                'video_codec': video_info['codec_name'] if video_info else None,
                'audio_codec': audio_info['codec_name'] if audio_info else None,
                'fps': eval(video_info['r_frame_rate']) if video_info and 'r_frame_rate' in video_info else None
            }
            
        except Exception as e:
            logging.error(f"Error getting video info: {e}")
            raise Exception(f"Error getting video info: {e}")
