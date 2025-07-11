import logging
from typing import List, Dict, Optional
from deep_translator import GoogleTranslator
from app.core.config import settings

class TranslationService:
    def __init__(self):
        self.supported_languages = settings.SUPPORTED_LANGUAGES
    
    async def translate_segments(self,
                               segments: List[Dict],
                               target_language: str,
                               source_language: str = 'auto') -> List[Dict]:
        """Traduce segmentos manteniendo timestamps"""
        try:
            logging.info(f"Translating {len(segments)} segments from {source_language} to {target_language}")
            
            if target_language not in self.supported_languages:
                raise Exception(f"Unsupported target language: {target_language}")
            
            # Crear traductor
            translator = GoogleTranslator(
                source=source_language,
                target=target_language
            )
            
            translated_segments = []
            
            for i, segment in enumerate(segments):
                original_text = segment.get('text', '').strip()
                
                if not original_text:
                    translated_segments.append(segment)
                    continue
                
                try:
                    # Traducir texto
                    translated_text = translator.translate(original_text)
                    
                    # Crear segmento traducido manteniendo estructura original
                    translated_segment = segment.copy()
                    translated_segment['text'] = translated_text
                    translated_segment['original_text'] = original_text
                    
                    translated_segments.append(translated_segment)
                    
                    if (i + 1) % 10 == 0:
                        logging.info(f"Translated {i + 1}/{len(segments)} segments")
                
                except Exception as e:
                    logging.warning(f"Translation failed for segment {i}: {e}")
                    # Mantener texto original si la traducción falla
                    translated_segments.append(segment)
            
            logging.info(f"Translation completed: {len(translated_segments)} segments")
            return translated_segments
            
        except Exception as e:
            logging.error(f"Translation error: {e}")
            raise Exception(f"Translation failed: {e}")
    
    async def translate_text(self, text: str, target_language: str, source_language: str = 'auto') -> str:
        """Traduce un texto simple"""
        try:
            if not text.strip():
                return text
            
            translator = GoogleTranslator(
                source=source_language,
                target=target_language
            )
            
            return translator.translate(text)
            
        except Exception as e:
            logging.error(f"Text translation error: {e}")
            return text  # Retornar texto original si falla
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Retorna idiomas soportados"""
        return self.supported_languages
