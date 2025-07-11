"""
Pruebas de integración para el flujo completo de transcripción.

Este módulo contiene pruebas que verifican la integración entre múltiples
componentes y el flujo completo con archivos reales.
"""

import os
import pytest
import tempfile
import shutil
from unittest.mock import patch
from fastapi.testclient import TestClient

from app.main import app
from app.services.audio_extractor import AudioExtractor
from app.services.transcription_service import TranscriptionService
from app.services.quality_evaluator import QualityEvaluator
from app.services.subtitle_generator import SubtitleGenerator
from app.utils.file_handler import FileHandler


class TestIntegrationFlow:
    """Pruebas de integración para el flujo completo."""
    
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
    
    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_real_video_transcription_short(self, test_video_path, temp_dir):
        """
        Prueba de integración con video real (versión corta).
        
        Esta prueba procesa solo los primeros segundos del video para velocidad.
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
            print(f"Audio extraído exitosamente.")
            
            # 2. Transcripción (limitada a modelo tiny para velocidad)
            print("Iniciando transcripción con modelo tiny...")
            transcription_result = await transcription_service.transcribe_audio(
                audio_path, "tiny", "auto"
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
            
            # Verificar que contiene elementos SRT válidos
            assert '-->' in srt_content
            assert '1\n' in srt_content
            
        except Exception as e:
            pytest.fail(f"Prueba de integración con video real falló: {str(e)}")
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_complete_flow_with_mocked_heavy_services(self, temp_dir):
        """
        Prueba de integración con servicios pesados mockeados.
        
        Esta prueba verifica la integración entre componentes sin la carga
        de procesamiento real de video/audio.
        """
        # Servicios reales para componentes ligeros
        quality_evaluator = QualityEvaluator()
        subtitle_generator = SubtitleGenerator()
        
        # Crear archivos mock
        video_path = os.path.join(temp_dir, "test_video.mp4")
        audio_path = os.path.join(temp_dir, "extracted_audio.wav")
        srt_path = os.path.join(temp_dir, "subtitles.srt")
        
        with open(video_path, 'wb') as f:
            f.write(b"mock video content")
        
        # Mock de servicios pesados
        with patch('app.services.audio_extractor.AudioExtractor.extract_audio') as mock_extract, \
             patch('app.services.transcription_service.TranscriptionService.transcribe_audio') as mock_transcribe:
            
            from app.models.schemas import VideoInfo
            
            # Configurar mocks
            mock_extract.return_value = VideoInfo(
                duration=30.0,
                size=5242880,
                video_codec="h264",
                audio_codec="aac",
                fps=30.0
            )
            
            mock_transcribe.return_value = {
                'text': 'This is a complete integration test with mocked heavy services.',
                'segments': [
                    {
                        'id': 0,
                        'start': 0.0,
                        'end': 10.0,
                        'text': 'This is a complete integration test',
                        'confidence': 0.93
                    },
                    {
                        'id': 1,
                        'start': 10.0,
                        'end': 20.0,
                        'text': ' with mocked heavy services.',
                        'confidence': 0.89
                    },
                    {
                        'id': 2,
                        'start': 20.0,
                        'end': 30.0,
                        'text': ' Testing integration flow.',
                        'confidence': 0.91
                    }
                ],
                'language': 'en'
            }
            
            # Ejecutar flujo
            audio_extractor = AudioExtractor()
            transcription_service = TranscriptionService()
            
            # 1. Extracción de audio
            video_info = await audio_extractor.extract_audio(video_path, audio_path)
            
            # 2. Transcripción
            transcription_result = await transcription_service.transcribe_audio(
                audio_path, "base", "auto"
            )
            
            # 3. Evaluación de calidad (servicio real)
            quality_report = quality_evaluator.evaluate_transcription(
                transcription_result, video_info.duration
            )
            
            # 4. Generación de SRT (servicio real)
            subtitle_generator.generate_srt(transcription_result['segments'], srt_path)
            
            # Verificaciones
            assert isinstance(video_info, VideoInfo)
            assert video_info.duration == 30.0
            assert transcription_result['text'] is not None
            assert len(transcription_result['segments']) == 3
            assert quality_report.overall_score > 0
            assert os.path.exists(srt_path)
            
            # Verificar SRT generado
            with open(srt_path, 'r', encoding='utf-8') as f:
                srt_content = f.read()
            
            assert 'This is a complete integration test' in srt_content
            assert '1\n' in srt_content
            assert '2\n' in srt_content
            assert '3\n' in srt_content
    
    @pytest.mark.integration
    def test_api_endpoints_integration(self, client):
        """
        Prueba de integración de endpoints API.
        
        Verifica que los endpoints responden correctamente y mantienen
        consistencia en las respuestas.
        """
        # 1. Health check
        health_response = client.get("/api/v1/health")
        assert health_response.status_code == 200
        health_data = health_response.json()
        assert "status" in health_data
        assert "services" in health_data
        
        # 2. Languages endpoint
        languages_response = client.get("/api/v1/languages")
        assert languages_response.status_code == 200
        languages_data = languages_response.json()
        assert isinstance(languages_data, dict)
        assert "auto" in languages_data
        assert "en" in languages_data
        
        # 3. Info endpoint
        info_response = client.get("/info")
        assert info_response.status_code == 200
        info_data = info_response.json()
        assert "api_title" in info_data
        assert "version" in info_data
    
    @pytest.mark.integration
    def test_file_validation_integration(self, client):
        """
        Prueba de integración de validación de archivos.
        
        Verifica que la validación funciona correctamente en el contexto
        de la API completa.
        """
        # Archivo con extensión inválida
        invalid_file_response = client.post(
            "/api/v1/transcribe",
            files={"file": ("test.txt", b"not a video", "text/plain")},
            headers={"Authorization": "Bearer dev_api_key_12345"}
        )
        assert invalid_file_response.status_code == 422
        
        # Sin autenticación
        no_auth_response = client.post(
            "/api/v1/transcribe",
            files={"file": ("test.mp4", b"mock video", "video/mp4")}
        )
        assert no_auth_response.status_code == 403
        
        # API key inválida
        invalid_key_response = client.post(
            "/api/v1/transcribe",
            files={"file": ("test.mp4", b"mock video", "video/mp4")},
            headers={"Authorization": "Bearer invalid_key"}
        )
        assert invalid_key_response.status_code == 401
    
    @pytest.mark.integration
    def test_job_lifecycle_integration(self, client):
        """
        Prueba de integración del ciclo de vida de trabajos.
        
        Verifica crear trabajo -> consultar estado -> obtener resultado.
        """
        with patch('app.utils.validators.FileValidator.validate_file_extension', return_value=True), \
             patch('app.utils.validators.FileValidator.validate_file_size', return_value=True), \
             patch('app.api.endpoints.transcription.process_video_transcription') as mock_process:
            
            # 1. Crear trabajo
            create_response = client.post(
                "/api/v1/transcribe",
                files={"file": ("test.mp4", b"mock video content", "video/mp4")},
                data={"language": "auto", "model_size": "base"},
                headers={"Authorization": "Bearer dev_api_key_12345"}
            )
            
            assert create_response.status_code == 200
            create_data = create_response.json()
            assert "job_id" in create_data
            job_id = create_data["job_id"]
            
            # 2. Consultar estado
            status_response = client.get(
                f"/api/v1/transcribe/{job_id}/status",
                headers={"Authorization": "Bearer dev_api_key_12345"}
            )
            
            assert status_response.status_code == 200
            status_data = status_response.json()
            assert status_data["job_id"] == job_id
            assert "status" in status_data
            
            # 3. Trabajo inexistente
            not_found_response = client.get(
                "/api/v1/transcribe/nonexistent-job/status",
                headers={"Authorization": "Bearer dev_api_key_12345"}
            )
            
            assert not_found_response.status_code == 404


if __name__ == "__main__":
    pytest.main(["-v", "tests/test_integration.py", "-m", "integration"])