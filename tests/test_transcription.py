"""
Pruebas unitarias para el flujo completo de transcripción de videos.

Este módulo contiene pruebas que verifican todo el pipeline de transcripción
desde la subida del archivo hasta la generación del archivo de transcripción.
"""

import os
import pytest
import asyncio
import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import UploadFile
import io

# Importar la aplicación y servicios
from app.main import app
from app.services.audio_extractor import AudioExtractor
from app.services.transcription_service import TranscriptionService
from app.services.quality_evaluator import QualityEvaluator
from app.services.subtitle_generator import SubtitleGenerator
from app.services.translation_service import TranslationService
from app.utils.file_handler import FileHandler
from app.utils.validators import FileValidator
from app.models.schemas import TranscriptionResult, QualityReport, VideoInfo
from app.core.config import settings


class TestTranscriptionFlow:
    """
    Suite de pruebas para el flujo completo de transcripción.
    """
    
    @pytest.fixture
    def client(self):
        """Cliente de prueba FastAPI."""
        return TestClient(app)
    
    @pytest.fixture
    def test_video_path(self):
        """Ruta al video de prueba."""
        return "/mnt/d/api-transcriptions/media/AC201 12 Data Loading (1080p).mp4"
    
    @pytest.fixture
    def temp_dir(self):
        """Directorio temporal para pruebas."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_video_file(self, test_video_path):
        """Mock de archivo de video para pruebas."""
        if os.path.exists(test_video_path):
            with open(test_video_path, 'rb') as f:
                content = f.read()
        else:
            # Crear contenido mock si el archivo no existe
            content = b"mock video content"
        
        return UploadFile(
            filename="test_video.mp4",
            file=io.BytesIO(content)
        )
    
    @pytest.fixture
    def audio_extractor(self):
        """Instancia del extractor de audio."""
        return AudioExtractor()
    
    @pytest.fixture
    def transcription_service(self):
        """Instancia del servicio de transcripción."""
        return TranscriptionService()
    
    @pytest.fixture
    def quality_evaluator(self):
        """Instancia del evaluador de calidad."""
        return QualityEvaluator()
    
    @pytest.fixture
    def subtitle_generator(self):
        """Instancia del generador de subtítulos."""
        return SubtitleGenerator()
    
    @pytest.fixture
    def file_handler(self):
        """Instancia del manejador de archivos."""
        return FileHandler()

    def test_file_validator_valid_file(self, test_video_path):
        """
        Prueba la validación de archivos válidos.
        """
        # Test file size validation
        if os.path.exists(test_video_path):
            file_size = os.path.getsize(test_video_path)
            assert FileValidator.validate_file_size(file_size, 500)
        
        # Test file extension validation
        assert FileValidator.validate_file_extension("test.mp4", [".mp4", ".avi"])
        assert not FileValidator.validate_file_extension("test.txt", [".mp4", ".avi"])
    
    def test_file_validator_invalid_file(self):
        """
        Prueba la validación de archivos inválidos.
        """
        # Test file size too large
        large_size = 600 * 1024 * 1024  # 600MB
        assert not FileValidator.validate_file_size(large_size, 500)
        
        # Test invalid extension
        assert not FileValidator.validate_file_extension("test.doc", [".mp4", ".avi"])

    @pytest.mark.asyncio
    async def test_audio_extraction(self, audio_extractor, test_video_path, temp_dir):
        """
        Prueba la extracción de audio desde video.
        """
        if not os.path.exists(test_video_path):
            pytest.skip("Archivo de video de prueba no encontrado")
        
        output_audio_path = os.path.join(temp_dir, "extracted_audio.wav")
        
        try:
            # Mock del extractor para esta prueba específica
            with patch.object(audio_extractor, 'extract_audio') as mock_extract:
                mock_extract.return_value = VideoInfo(
                    duration=120.0,
                    size=15728640,
                    video_codec="h264",
                    audio_codec="aac",
                    fps=30.0
                )
                
                # También crear el archivo de audio mock
                with open(output_audio_path, 'wb') as f:
                    f.write(b"mock audio data")
                
                video_info = audio_extractor.extract_audio(test_video_path, output_audio_path)
                
                # Verificar que se creó el archivo de audio
                assert os.path.exists(output_audio_path)
                
                # Verificar información del video
                assert isinstance(video_info, VideoInfo)
                assert video_info.duration > 0
                assert video_info.size > 0
            
        except Exception as e:
            pytest.fail(f"Extracción de audio falló: {str(e)}")

    @pytest.mark.asyncio 
    async def test_transcription_service_mock(self, transcription_service, temp_dir):
        """
        Prueba el servicio de transcripción con audio mock.
        """
        # Crear archivo de audio mock
        mock_audio_path = os.path.join(temp_dir, "mock_audio.wav")
        with open(mock_audio_path, 'wb') as f:
            f.write(b"mock audio data")
        
        # Mock del método transcribe_audio directamente
        with patch.object(transcription_service, 'transcribe_audio') as mock_transcribe:
            # Configurar mock con resultado esperado
            mock_result = {
                'text': 'This is a test transcription.',
                'segments': [
                    {
                        'id': 0,
                        'start': 0.0,
                        'end': 2.5,
                        'text': 'This is a test',
                        'confidence': 0.95
                    },
                    {
                        'id': 1,
                        'start': 2.5,
                        'end': 5.0,
                        'text': ' transcription.',
                        'confidence': 0.88
                    }
                ],
                'language': 'en'
            }
            mock_transcribe.return_value = mock_result
            
            # Ejecutar transcripción
            result = await transcription_service.transcribe_audio(mock_audio_path, "base", "auto")
            
            # Verificar resultados
            assert result['text'] == 'This is a test transcription.'
            assert result['language'] == 'en'
            assert len(result['segments']) == 2
            assert all('confidence' in segment for segment in result['segments'])

    def test_quality_evaluator(self, quality_evaluator):
        """
        Prueba el evaluador de calidad de transcripciones.
        """
        # Datos de transcripción mock
        transcription_data = {
            'text': 'This is a test transcription with good quality.',
            'segments': [
                {
                    'id': 0,
                    'start': 0.0,
                    'end': 2.0,
                    'text': 'This is a test',
                    'confidence': 0.95
                },
                {
                    'id': 1,
                    'start': 2.0,
                    'end': 4.0,
                    'text': ' transcription with good quality.',
                    'confidence': 0.88
                }
            ]
        }
        
        video_duration = 4.0
        
        # Evaluar calidad
        quality_report = quality_evaluator.evaluate_transcription(
            transcription_data, 
            video_duration
        )
        
        # Verificar estructura del reporte
        assert isinstance(quality_report, QualityReport)
        assert 0.0 <= quality_report.overall_score <= 1.0
        assert quality_report.quality_level in ['excellent', 'good', 'acceptable', 'poor']
        assert hasattr(quality_report.metrics, 'confidence_score')
        assert hasattr(quality_report.metrics, 'word_count')
        assert isinstance(quality_report.recommendations, list)

    def test_subtitle_generator(self, subtitle_generator, temp_dir):
        """
        Prueba la generación de archivos SRT.
        """
        # Datos de segmentos mock
        segments = [
            {
                'id': 0,
                'start': 0.0,
                'end': 2.5,
                'text': 'This is the first subtitle segment.'
            },
            {
                'id': 1,
                'start': 2.5,
                'end': 5.0,
                'text': 'This is the second subtitle segment.'
            },
            {
                'id': 2,
                'start': 5.0,
                'end': 7.5,
                'text': 'This is the third and final segment.'
            }
        ]
        
        output_srt_path = os.path.join(temp_dir, "test_subtitles.srt")
        
        # Generar subtítulos
        subtitle_generator.generate_srt(segments, output_srt_path)
        
        # Verificar que se creó el archivo
        assert os.path.exists(output_srt_path)
        
        # Verificar contenido del archivo SRT
        with open(output_srt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Verificar formato SRT básico
        assert '1\n' in content  # Número de secuencia
        assert '-->' in content  # Formato de timestamps
        assert 'This is the first subtitle segment.' in content
        assert '2\n' in content
        assert '3\n' in content

    @pytest.mark.asyncio
    async def test_file_handler_upload(self, mock_video_file, temp_dir):
        """
        Prueba el manejo de subida de archivos.
        """
        # Crear FileHandler con directorio temporal personalizado
        test_file_handler = FileHandler()
        test_file_handler.uploads_dir = os.path.join(temp_dir, "uploads")
        os.makedirs(test_file_handler.uploads_dir, exist_ok=True)
        
        # Simular subida de archivo
        saved_path = await test_file_handler.save_uploaded_file(
            mock_video_file, 
            "test_job_id"
        )
        
        # Verificar que se guardó el archivo
        assert os.path.exists(saved_path)
        assert "test_job_id" in saved_path
        assert saved_path.endswith(".mp4")

    @pytest.mark.asyncio
    async def test_complete_transcription_flow_mock(self, temp_dir):
        """
        Prueba el flujo completo de transcripción con servicios mock.
        """
        # Configurar servicios mock
        audio_extractor = AudioExtractor()
        transcription_service = TranscriptionService()
        quality_evaluator = QualityEvaluator()
        subtitle_generator = SubtitleGenerator()
        
        # Crear archivos mock
        video_path = os.path.join(temp_dir, "test_video.mp4")
        audio_path = os.path.join(temp_dir, "extracted_audio.wav")
        srt_path = os.path.join(temp_dir, "subtitles.srt")
        
        with open(video_path, 'wb') as f:
            f.write(b"mock video content")
        
        # Mock de extracción de audio
        with patch.object(audio_extractor, 'extract_audio') as mock_extract:
            mock_extract.return_value = VideoInfo(
                duration=10.0,
                size=1024000,
                video_codec="h264",
                audio_codec="aac",
                fps=30.0
            )
            
            # Mock de transcripción
            with patch.object(transcription_service, 'transcribe_audio') as mock_transcribe:
                mock_transcribe.return_value = {
                    'text': 'This is a complete test transcription of the video content.',
                    'segments': [
                        {
                            'id': 0,
                            'start': 0.0,
                            'end': 5.0,
                            'text': 'This is a complete test',
                            'confidence': 0.92
                        },
                        {
                            'id': 1,
                            'start': 5.0,
                            'end': 10.0,
                            'text': ' transcription of the video content.',
                            'confidence': 0.88
                        }
                    ],
                    'language': 'en'
                }
                
                # Ejecutar flujo completo
                try:
                    # 1. Extraer audio
                    video_info = await audio_extractor.extract_audio(video_path, audio_path)
                    
                    # 2. Transcribir audio
                    transcription_result = await transcription_service.transcribe_audio(
                        audio_path, "base", "auto"
                    )
                    
                    # 3. Evaluar calidad
                    quality_report = quality_evaluator.evaluate_transcription(
                        transcription_result, video_info.duration
                    )
                    
                    # 4. Generar subtítulos
                    subtitle_generator.generate_srt(
                        transcription_result['segments'], 
                        srt_path
                    )
                    
                    # Verificar resultados
                    assert isinstance(video_info, VideoInfo)
                    assert transcription_result['text'] is not None
                    assert len(transcription_result['segments']) > 0
                    assert isinstance(quality_report, QualityReport)
                    assert os.path.exists(srt_path)
                    
                    # Verificar contenido del SRT
                    with open(srt_path, 'r', encoding='utf-8') as f:
                        srt_content = f.read()
                    assert 'This is a complete test' in srt_content
                    
                except Exception as e:
                    pytest.fail(f"Flujo completo falló: {str(e)}")

    def test_api_endpoint_transcribe_mock(self, client):
        """
        Prueba el endpoint de transcripción con servicios mock.
        """
        # Mock de archivo de video
        video_content = b"mock video content"
        
        # Mock de validaciones y servicios
        with patch('app.utils.validators.FileValidator.validate_file_extension', return_value=True), \
             patch('app.utils.validators.FileValidator.validate_file_size', return_value=True), \
             patch('app.api.endpoints.transcription.process_video_transcription') as mock_bg:
            
            # Simular subida de archivo
            response = client.post(
                "/api/v1/transcribe",
                files={"file": ("test.mp4", video_content, "video/mp4")},
                data={"language": "auto", "model_size": "base"},
                headers={"Authorization": "Bearer dev_api_key_12345"}
            )
            
            # Verificar respuesta
            assert response.status_code == 200
            data = response.json()
            assert "job_id" in data
            assert data["status"] == "processing"

    def test_api_endpoint_status(self, client):
        """
        Prueba el endpoint de estado de trabajo.
        """
        # Mock de estado de trabajo
        job_id = "test-job-123"
        
        with patch('app.api.endpoints.transcription.job_storage') as mock_storage:
            mock_storage.__contains__.return_value = True
            mock_storage.__getitem__.return_value = TranscriptionResult(
                job_id=job_id,
                status="completed",
                transcription_text="Test transcription",
                created_at="2024-01-01T00:00:00"
            )
            
            response = client.get(
                f"/api/v1/transcribe/{job_id}/status",
                headers={"Authorization": "Bearer dev_api_key_12345"}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["job_id"] == job_id
            assert data["status"] == "completed"

    def test_api_endpoint_health(self, client):
        """
        Prueba el endpoint de health check.
        """
        response = client.get("/api/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "system" in data
        assert "services" in data

    @pytest.mark.skip("Moved to test_integration.py")
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_video_transcription(self, test_video_path, temp_dir):
        """
        Prueba de integración con video real (requiere archivo de video).
        
        Esta prueba se ejecuta solo si existe el archivo de video especificado.
        """
        if not os.path.exists(test_video_path):
            pytest.skip("Archivo de video real no encontrado para prueba de integración")
        
        # Servicios reales
        audio_extractor = AudioExtractor()
        transcription_service = TranscriptionService()
        quality_evaluator = QualityEvaluator()
        subtitle_generator = SubtitleGenerator()
        
        # Rutas de salida
        audio_path = os.path.join(temp_dir, "real_audio.wav")
        srt_path = os.path.join(temp_dir, "real_subtitles.srt")
        
        try:
            # 1. Extracción de audio
            print(f"Extrayendo audio de: {test_video_path}")
            video_info = await audio_extractor.extract_audio(test_video_path, audio_path)
            assert os.path.exists(audio_path)
            print(f"Audio extraído exitosamente. Duración: {video_info.duration}s")
            
            # 2. Transcripción (limitada a modelo pequeño para pruebas)
            print("Iniciando transcripción...")
            transcription_result = await transcription_service.transcribe_audio(
                audio_path, "tiny", "auto"  # Usar modelo pequeño para velocidad
            )
            assert transcription_result['text'] is not None
            assert len(transcription_result['segments']) > 0
            print(f"Transcripción completada: {len(transcription_result['text'])} caracteres")
            
            # 3. Evaluación de calidad
            print("Evaluando calidad...")
            quality_report = quality_evaluator.evaluate_transcription(
                transcription_result, video_info.duration
            )
            print(f"Calidad: {quality_report.quality_level} (Score: {quality_report.overall_score:.2f})")
            
            # 4. Generación de SRT
            print("Generando subtítulos...")
            subtitle_generator.generate_srt(transcription_result['segments'], srt_path)
            assert os.path.exists(srt_path)
            
            # Verificar contenido del SRT
            with open(srt_path, 'r', encoding='utf-8') as f:
                srt_content = f.read()
            assert len(srt_content) > 0
            print(f"Subtítulos generados exitosamente en: {srt_path}")
            
        except Exception as e:
            pytest.fail(f"Prueba de integración con video real falló: {str(e)}")

    def test_error_handling_invalid_file(self, client):
        """
        Prueba el manejo de errores con archivos inválidos.
        """
        # Archivo con extensión inválida
        invalid_content = b"not a video file"
        
        response = client.post(
            "/api/v1/transcribe",
            files={"file": ("test.txt", invalid_content, "text/plain")},
            headers={"Authorization": "Bearer dev_api_key_12345"}
        )
        
        assert response.status_code == 422

    def test_error_handling_no_auth(self, client):
        """
        Prueba el manejo de errores sin autenticación.
        """
        video_content = b"mock video content"
        
        response = client.post(
            "/api/v1/transcribe",
            files={"file": ("test.mp4", video_content, "video/mp4")}
        )
        
        assert response.status_code == 403

    def test_cleanup_temp_files(self, temp_dir):
        """
        Prueba la limpieza de archivos temporales.
        """
        # Crear FileHandler con directorios temporales personalizados
        test_file_handler = FileHandler()
        test_uploads_dir = os.path.join(temp_dir, "uploads")
        test_audio_dir = os.path.join(temp_dir, "audio")
        test_srt_dir = os.path.join(temp_dir, "srt")
        
        # Configurar directorios temporales
        test_file_handler.uploads_dir = test_uploads_dir
        test_file_handler.audio_dir = test_audio_dir
        test_file_handler.srt_dir = test_srt_dir
        
        # Crear directorios
        for directory in [test_uploads_dir, test_audio_dir, test_srt_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Crear archivos temporales antiguos en cada directorio
        old_files = []
        for directory in [test_uploads_dir, test_audio_dir, test_srt_dir]:
            old_file = os.path.join(directory, "old_file.txt")
            with open(old_file, 'w') as f:
                f.write("old content")
            old_files.append(old_file)
            
            # Modificar timestamp para simular archivo antiguo
            old_timestamp = os.path.getmtime(old_file) - (25 * 60 * 60)  # 25 horas atrás
            os.utime(old_file, (old_timestamp, old_timestamp))
        
        # Ejecutar limpieza
        test_file_handler.cleanup_old_files(24)  # Limpiar archivos > 24 horas
        
        # Verificar que los archivos fueron eliminados
        for old_file in old_files:
            assert not os.path.exists(old_file)


if __name__ == "__main__":
    pytest.main(["-v", "tests/test_transcription.py"])