import numpy as np
import wave
import tempfile
import os
import asyncio
import sys
import logging

# Agregar el directorio del proyecto al path
sys.path.append(r'd:\Desarrollo\api-transcriptions')

from app.services.realtime_transcription_service_v4 import RealTimeTranscriptionServiceV4

# Configurar logging
logging.basicConfig(level=logging.INFO)

def create_test_audio():
    """Crea un archivo de audio de prueba con tono"""
    # Generar 3 segundos de tono de 440Hz (nota La)
    sample_rate = 16000
    duration = 3.0
    samples = int(sample_rate * duration)
    
    t = np.linspace(0, duration, samples, False)
    # Tono de 440Hz con volumen moderado
    audio = 0.3 * np.sin(2 * np.pi * 440 * t)
    
    # Convertir a int16
    audio_int16 = (audio * 32767).astype(np.int16)
    
    # Guardar como WAV
    temp_wav = os.path.join(tempfile.gettempdir(), "test_tone.wav")
    with wave.open(temp_wav, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)  # 16kHz
        wav_file.writeframes(audio_int16.tobytes())
    
    print(f"Archivo de audio de prueba creado: {temp_wav}")
    return temp_wav, audio_int16.tobytes()

def create_real_speech_audio():
    """Crea un archivo con síntesis simple de voz"""
    sample_rate = 16000
    duration = 3.0
    samples = int(sample_rate * duration)
    
    # Simular formantes de la vocal "a" (española)
    # Formantes aproximados: F1=730Hz, F2=1090Hz, F3=2440Hz
    t = np.linspace(0, duration, samples, False)
    
    # Fundamental frequency (pitch) - 150Hz para voz masculina
    f0 = 150
    
    # Crear formantes
    formant1 = 0.4 * np.sin(2 * np.pi * 730 * t)
    formant2 = 0.3 * np.sin(2 * np.pi * 1090 * t)
    formant3 = 0.2 * np.sin(2 * np.pi * 2440 * t)
    
    # Modulación de la fundamental
    carrier = np.sin(2 * np.pi * f0 * t)
    
    # Combinar formantes con la fundamental
    speech = carrier * (formant1 + formant2 + formant3)
    
    # Aplicar envelope para hacer más natural
    envelope = np.exp(-2 * t) * np.sin(np.pi * t / duration)
    speech = speech * envelope * 0.3
    
    # Convertir a int16
    audio_int16 = (speech * 32767).astype(np.int16)
    
    # Guardar como WAV
    temp_wav = os.path.join(tempfile.gettempdir(), "test_speech.wav")
    with wave.open(temp_wav, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)  # 16kHz
        wav_file.writeframes(audio_int16.tobytes())
    
    print(f"Archivo de síntesis de voz creado: {temp_wav}")
    return temp_wav, audio_int16.tobytes()

async def test_whisper_direct():
    """Prueba Whisper directamente con archivos de audio conocidos"""
    import whisper
    
    print("=== PRUEBA DIRECTA DE WHISPER ===")
    
    # Cargar modelo
    print("Cargando modelo Whisper...")
    model = whisper.load_model("tiny")
    print("Modelo cargado!")
    
    # Probar con tono
    print("\n1. Probando con tono de 440Hz...")
    tone_file, _ = create_test_audio()
    result1 = model.transcribe(tone_file, language="es")
    print(f"Resultado tono: '{result1.get('text', '').strip()}'")
    os.remove(tone_file)
    
    # Probar con síntesis de voz
    print("\n2. Probando con síntesis de voz...")
    speech_file, _ = create_real_speech_audio()
    result2 = model.transcribe(speech_file, language="es")
    print(f"Resultado síntesis: '{result2.get('text', '').strip()}'")
    os.remove(speech_file)
    
    print("\n=== FIN PRUEBA DIRECTA ===")

async def test_service_with_synthetic():
    """Prueba el servicio con audio sintético"""
    print("\n=== PRUEBA DEL SERVICIO ===")
    
    service = RealTimeTranscriptionServiceV4()
    
    # Crear sesión
    session_id = "test_session_123"
    session = await service.create_session(
        session_id=session_id,
        language="es",
        model_size="tiny"
    )
    print(f"Sesión creada: {session.session_id}")
    
    # Crear audio de prueba
    _, audio_bytes = create_real_speech_audio()
    
    # Convertir a base64
    import base64
    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
    
    # Procesar chunk
    print("Procesando chunk de audio sintético...")
    response = await service.process_audio_chunk(
        session_id=session_id,
        chunk_id="test_chunk_1",
        audio_base64=audio_base64
    )
    
    if response:
        print(f"Transcripción: '{response.transcription}'")
        print(f"Idioma detectado: {response.detected_language}")
    else:
        print("No se obtuvo respuesta")
    
    print("=== FIN PRUEBA SERVICIO ===")

if __name__ == "__main__":
    print("DIAGNÓSTICO COMPLETO DE WHISPER\n")
    
    # Probar Whisper directamente
    asyncio.run(test_whisper_direct())
    
    # Probar el servicio
    asyncio.run(test_service_with_synthetic())
