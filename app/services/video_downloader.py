import os
import logging
import yt_dlp
import asyncio
import concurrent.futures
from typing import Optional, Dict, Any
from urllib.parse import urlparse

class VideoDownloaderService:
    """
    Servicio para descargar videos desde URLs (YouTube, Vimeo, etc.)
    usando yt-dlp que soporta múltiples plataformas
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.temp_dir = "temp/downloads"
        os.makedirs(self.temp_dir, exist_ok=True)
    
    def get_video_info(self, url: str) -> Dict[str, Any]:
        """
        Obtiene información del video sin descargarlo
        """
        try:
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                
                return {
                    'title': info.get('title', 'Unknown'),
                    'duration': info.get('duration', 0),
                    'uploader': info.get('uploader', 'Unknown'),
                    'view_count': info.get('view_count', 0),
                    'upload_date': info.get('upload_date', 'Unknown'),
                    'description': info.get('description', '')[:500] + '...' if info.get('description') else '',
                    'thumbnail': info.get('thumbnail'),
                    'webpage_url': info.get('webpage_url', url),
                    'extractor': info.get('extractor', 'Unknown'),
                    'formats_available': len(info.get('formats', [])),
                    'has_audio': any(f.get('acodec') != 'none' for f in info.get('formats', [])),
                    'has_video': any(f.get('vcodec') != 'none' for f in info.get('formats', []))
                }
        except Exception as e:
            self.logger.error(f"Error getting video info from {url}: {e}")
            raise ValueError(f"Could not extract video information: {str(e)}")
    
    def _is_valid_url(self, url: str) -> bool:
        """
        Valida si la URL es válida
        """
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False
    
    def _get_download_options(self, job_id: str, quality: str = "best") -> Dict[str, Any]:
        """
        Configuración de descarga para yt-dlp
        """
        output_path = os.path.join(self.temp_dir, f"{job_id}.%(ext)s")
        
        # Opciones de calidad
        quality_options = {
            "best": "best[height<=1080]",
            "medium": "best[height<=720]", 
            "low": "best[height<=480]",
            "audio_only": "bestaudio/best"
        }
        
        return {
            'format': quality_options.get(quality, "best[height<=1080]"),
            'outtmpl': output_path,
            'writeinfojson': True,
            'writesubtitles': False,
            'writeautomaticsub': False,
            'ignoreerrors': False,
            'no_warnings': True,
            'extractaudio': False,
            'audioformat': 'mp3',
            'embed_subs': False,
            # Limitar tamaño de archivo (500MB)
            'format_sort': ['filesize:500M'],
            # Headers para evitar bloqueos
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
        }
    
    async def download_video(self, url: str, quality: str = "medium", job_id: str = None) -> Dict[str, Any]:
        """
        Descarga video desde URL de forma asíncrona
        Retorna diccionario con 'success', 'file_path', 'error'
        """
        if not self._is_valid_url(url):
            return {
                'success': False,
                'file_path': None,
                'error': 'Invalid URL provided'
            }
        
        # Generar job_id si no se proporciona
        if not job_id:
            import uuid
            job_id = str(uuid.uuid4())
        
        try:
            # Verificar que el video existe y es accesible
            video_info = self.get_video_info(url)
            self.logger.info(f"Starting download for: {video_info.get('title', 'Unknown')}")
            
            # Configurar opciones de descarga
            ydl_opts = self._get_download_options(job_id, quality)
            
            # Ejecutar descarga en un hilo separado para no bloquear
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                downloaded_file = await loop.run_in_executor(
                    executor, 
                    self._download_sync, 
                    url, 
                    ydl_opts
                )
            
            # Verificar múltiples posibles ubicaciones del archivo
            possible_files = []
            if downloaded_file:
                possible_files.append(downloaded_file)
            
            # También buscar por patrón de nombre
            import glob
            pattern = os.path.join(self.temp_dir, f"{job_id}.*")
            glob_files = glob.glob(pattern)
            possible_files.extend(glob_files)
            
            # Filtrar archivos que no sean .info.json
            actual_files = [f for f in possible_files if f and not f.endswith('.info.json') and os.path.exists(f)]
            
            if not actual_files:
                # Último intento: buscar cualquier archivo reciente en el directorio
                import time
                recent_files = []
                for file in os.listdir(self.temp_dir):
                    file_path = os.path.join(self.temp_dir, file)
                    if (os.path.isfile(file_path) and 
                        not file.endswith('.info.json') and
                        time.time() - os.path.getctime(file_path) < 60):  # Archivo creado en el último minuto
                        recent_files.append(file_path)
                
                if recent_files:
                    # Tomar el archivo más reciente
                    downloaded_file = max(recent_files, key=os.path.getctime)
                else:
                    raise Exception("Download completed but file not found")
            else:
                downloaded_file = actual_files[0]
            
            if not os.path.exists(downloaded_file):
                raise Exception("Downloaded file path exists but file not accessible")
            
            self.logger.info(f"Successfully downloaded: {downloaded_file}")
            return {
                'success': True,
                'file_path': downloaded_file,
                'error': None
            }
            
        except Exception as e:
            self.logger.error(f"Error downloading video from {url}: {e}")
            # Limpiar archivos parciales en caso de error
            self._cleanup_partial_files(job_id)
            return {
                'success': False,
                'file_path': None,
                'error': str(e)
            }
    
    def _download_sync(self, url: str, ydl_opts: Dict[str, Any]) -> str:
        """
        Función síncrona para la descarga real
        """
        downloaded_file = None
        
        # Crear una copia de las opciones para no modificar el original
        opts = ydl_opts.copy()
        
        with yt_dlp.YoutubeDL(opts) as ydl:
            # Hook para capturar el archivo descargado
            def progress_hook(d):
                nonlocal downloaded_file
                if d['status'] == 'finished':
                    downloaded_file = d['filename']
                    self.logger.info(f"Download finished: {downloaded_file}")
                elif d['status'] == 'downloading':
                    # También capturar el nombre del archivo durante la descarga
                    if 'filename' in d:
                        downloaded_file = d['filename']
            
            opts['progress_hooks'] = [progress_hook]
            
            try:
                ydl.download([url])
            except Exception as e:
                self.logger.error(f"yt-dlp download error: {e}")
                # Intentar encontrar el archivo aunque haya error
                pass
        
        # Si no se capturó el archivo por hook, intentar buscar manualmente
        if not downloaded_file:
            self.logger.warning("Hook didn't capture filename, searching manually...")
            # Obtener el template de salida y buscar archivos que coincidan
            outtmpl = ydl_opts.get('outtmpl', '')
            if outtmpl:
                import glob
                # Reemplazar %(ext)s con *
                pattern = outtmpl.replace('%(ext)s', '*')
                matches = glob.glob(pattern)
                if matches:
                    # Filtrar archivos que no sean .info.json
                    video_files = [f for f in matches if not f.endswith('.info.json')]
                    if video_files:
                        downloaded_file = video_files[0]
                        self.logger.info(f"Found file manually: {downloaded_file}")
        
        return downloaded_file
    
    def _cleanup_partial_files(self, job_id: str):
        """
        Limpia archivos parciales en caso de error
        """
        try:
            import glob
            pattern = os.path.join(self.temp_dir, f"{job_id}.*")
            files = glob.glob(pattern)
            for file in files:
                if os.path.exists(file):
                    os.remove(file)
                    self.logger.info(f"Cleaned up partial file: {file}")
        except Exception as e:
            self.logger.warning(f"Error cleaning up partial files: {e}")
    
    def cleanup_downloaded_file(self, file_path: str):
        """
        Limpia archivo descargado después del procesamiento
        """
        try:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
                self.logger.info(f"Cleaned up downloaded file: {file_path}")
                
                # También limpiar archivo .info.json si existe
                info_file = file_path.rsplit('.', 1)[0] + '.info.json'
                if os.path.exists(info_file):
                    os.remove(info_file)
                    self.logger.info(f"Cleaned up info file: {info_file}")
        except Exception as e:
            self.logger.warning(f"Error cleaning up downloaded file {file_path}: {e}")
    
    def get_supported_sites(self) -> list:
        """
        Obtiene lista de sitios soportados por yt-dlp
        """
        popular_sites = [
            'YouTube', 'Vimeo', 'Dailymotion', 'Facebook', 'Instagram',
            'Twitter', 'TikTok', 'Twitch', 'Reddit', 'SoundCloud',
            'BBC iPlayer', 'CNN', 'ESPN', 'Coursera', 'Udemy'
        ]
        return popular_sites
    
    def validate_url_accessibility(self, url: str) -> Dict[str, Any]:
        """
        Valida que la URL sea accesible y contenga video/audio
        """
        try:
            info = self.get_video_info(url)
            
            validation_result = {
                'valid': True,
                'accessible': True,
                'has_video': info.get('has_video', False),
                'has_audio': info.get('has_audio', False),
                'duration': info.get('duration', 0),
                'title': info.get('title', 'Unknown'),
                'extractor': info.get('extractor', 'Unknown'),
                'warnings': [],
                'errors': []
            }
            
            # Validaciones adicionales
            if info.get('duration', 0) > 7200:  # 2 horas
                validation_result['warnings'].append("Video is longer than 2 hours, processing may take significant time")
            
            if not info.get('has_audio'):
                validation_result['errors'].append("Video does not contain audio track")
                validation_result['valid'] = False
            
            if info.get('duration', 0) == 0:
                validation_result['warnings'].append("Could not determine video duration")
            
            return validation_result
            
        except Exception as e:
            return {
                'valid': False,
                'accessible': False,
                'has_video': False,
                'has_audio': False,
                'duration': 0,
                'title': 'Unknown',
                'extractor': 'Unknown',
                'warnings': [],
                'errors': [f"URL validation failed: {str(e)}"]
            }
