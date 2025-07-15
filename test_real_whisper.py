#!/usr/bin/env python3
"""
Test Definitivo - Whisper con Archivo Real
Este script graba 5 segundos de audio real y los transcribe
"""

import whisper
import sounddevice as sd
import numpy as np
import scipy.io.wavfile
import tempfile
import os
import time
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def record_real_audio(duration=5, sample_rate=16000):
    """Graba audio real del micrófono"""
    print(f"\n🎤 HABLA AHORA por {duration} segundos...")
    print("Di algo como: 'Hola, mi nombre es Juan y estoy probando la transcripción'")
    print("Cuenta regresiva: ", end="", flush=True)
    
    for i in range(3, 0, -1):
        print(f"{i}... ", end="", flush=True)
        time.sleep(1)
    
    print("¡GRABANDO!")
    
    # Grabar audio
    audio_data = sd.rec(int(duration * sample_rate), 
                       samplerate=sample_rate, 
                       channels=1, 
                       dtype=np.float32)
    sd.wait()  # Esperar que termine la grabación
    
    print("✅ Grabación completada")
    
    # Convertir a formato int16 para WAV
    audio_int16 = (audio_data * 32767).astype(np.int16)
    
    return audio_int16, sample_rate

def test_whisper_with_real_audio():
    """Test principal con audio real"""
    print("=" * 60)
    print("🚀 TEST DEFINITIVO - WHISPER CON AUDIO REAL")
    print("=" * 60)
    
    try:
        # 1. Grabar audio real
        print("\n1. Grabando audio real...")
        audio_data, sample_rate = record_real_audio(duration=5)
        
        # 2. Guardar como WAV
        temp_dir = tempfile.gettempdir()
        wav_file = os.path.join(temp_dir, f"test_real_{int(time.time())}.wav")
        
        scipy.io.wavfile.write(wav_file, sample_rate, audio_data)
        print(f"✅ Audio guardado: {wav_file}")
        
        # Verificar archivo
        file_size = os.path.getsize(wav_file)
        duration = len(audio_data) / sample_rate
        print(f"📊 Archivo: {file_size} bytes, {duration:.2f}s")
        
        # 3. Cargar modelos Whisper
        print("\n2. Cargando modelos Whisper...")
        
        models = ["tiny", "base"]
        for model_name in models:
            print(f"\n--- Probando modelo: {model_name} ---")
            
            try:
                model = whisper.load_model(model_name)
                print(f"✅ Modelo {model_name} cargado")
                
                # Transcribir con diferentes configuraciones
                configs = [
                    {
                        "name": "Configuración Estándar",
                        "params": {
                            "language": "es",
                            "task": "transcribe",
                            "temperature": 0.0,
                            "no_speech_threshold": 0.6
                        }
                    },
                    {
                        "name": "Configuración Optimizada",
                        "params": {
                            "language": "es", 
                            "task": "transcribe",
                            "temperature": 0.2,
                            "no_speech_threshold": 0.4,
                            "prompt": "Transcribe el siguiente audio en español:"
                        }
                    },
                    {
                        "name": "Auto-detección",
                        "params": {
                            "language": None,
                            "task": "transcribe", 
                            "temperature": 0.1,
                            "no_speech_threshold": 0.3
                        }
                    }
                ]
                
                for config in configs:
                    print(f"\n  🔧 {config['name']}:")
                    try:
                        result = model.transcribe(wav_file, **config['params'])
                        
                        text = result.get("text", "").strip()
                        language = result.get("language", "unknown")
                        
                        print(f"    🗣️  Idioma detectado: {language}")
                        print(f"    📝 Transcripción: '{text}'")
                        
                        if text:
                            print(f"    🎉 ¡ÉXITO! Whisper transcribió el audio")
                        else:
                            print(f"    ⚠️  Transcripción vacía")
                            
                    except Exception as e:
                        print(f"    ❌ Error: {e}")
                
            except Exception as e:
                print(f"❌ Error cargando modelo {model_name}: {e}")
        
        # 4. Limpiar
        if os.path.exists(wav_file):
            os.remove(wav_file)
            print(f"\n🧹 Archivo temporal eliminado")
            
    except Exception as e:
        print(f"❌ Error general: {e}")
        
    print("\n" + "=" * 60)
    print("🏁 Test completado")
    print("=" * 60)

if __name__ == "__main__":
    try:
        # Verificar dependencias
        import sounddevice
        print("✅ sounddevice disponible")
    except ImportError:
        print("❌ Instalando sounddevice...")
        os.system("pip install sounddevice")
        
    test_whisper_with_real_audio()
