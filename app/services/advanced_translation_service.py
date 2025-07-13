"""
Servicio de traducción avanzado con múltiples proveedores de alta calidad.

Este módulo proporciona traducción de alta calidad usando:
- DeepL API (calidad superior, especialmente para idiomas europeos)
- Microsoft Translator (cost-effective, 100+ idiomas)
- LibreTranslate (self-hosted, privacidad total)
- Google Translate (fallback)

Ejemplo de uso:
    service = AdvancedTranslationService()
    result = await service.translate_text("Hello world", "en", "es")
"""

import asyncio
import logging
import os
from typing import Optional, Dict, List, Any
import aiohttp
import json
from enum import Enum

class TranslationProvider(Enum):
    """Proveedores de traducción disponibles"""
    DEEPL = "deepl"
    MICROSOFT = "microsoft"
    LIBRETRANSLATE = "libretranslate"
    GOOGLE = "google"

class AdvancedTranslationService:
    """Servicio de traducción avanzado con múltiples proveedores"""
    
    def __init__(self):
        self.providers = {
            TranslationProvider.DEEPL: self._translate_deepl,
            TranslationProvider.MICROSOFT: self._translate_microsoft,
            TranslationProvider.LIBRETRANSLATE: self._translate_libretranslate,
            TranslationProvider.GOOGLE: self._translate_google
        }
        
        # Configuración de APIs
        self.deepl_api_key = os.getenv('DEEPL_API_KEY')
        self.microsoft_api_key = os.getenv('MICROSOFT_TRANSLATOR_KEY')
        self.microsoft_region = os.getenv('MICROSOFT_TRANSLATOR_REGION', 'global')
        self.libretranslate_url = os.getenv('LIBRETRANSLATE_URL', 'http://localhost:5000')
        
        # Mapeo de códigos de idioma para diferentes APIs
        self.language_mappings = {
            TranslationProvider.DEEPL: {
                'es': 'ES', 'en': 'EN', 'fr': 'FR', 'de': 'DE', 
                'it': 'IT', 'pt': 'PT-PT', 'ru': 'RU', 'zh': 'ZH',
                'ja': 'JA', 'ko': 'KO'
            },
            TranslationProvider.MICROSOFT: {
                'es': 'es', 'en': 'en', 'fr': 'fr', 'de': 'de',
                'it': 'it', 'pt': 'pt', 'ru': 'ru', 'zh': 'zh',
                'ja': 'ja', 'ko': 'ko', 'ar': 'ar', 'hi': 'hi'
            }
        }
        
        # Orden de prioridad de proveedores según calidad
        self.provider_priority = [
            TranslationProvider.DEEPL,
            TranslationProvider.MICROSOFT,
            TranslationProvider.LIBRETRANSLATE,
            TranslationProvider.GOOGLE
        ]
    
    async def translate_text(self, 
                           text: str, 
                           source_lang: str, 
                           target_lang: str,
                           preferred_provider: Optional[TranslationProvider] = None) -> Optional[str]:
        """
        Traduce texto usando el mejor proveedor disponible
        
        Args:
            text: Texto a traducir
            source_lang: Idioma de origen
            target_lang: Idioma destino
            preferred_provider: Proveedor preferido (opcional)
            
        Returns:
            Texto traducido o None si falla
        """
        if not text or not text.strip():
            return text
            
        # Si no hay cambio de idioma, retornar original
        if source_lang == target_lang:
            return text
            
        logging.info(f"Advanced translation: '{text[:50]}...' from {source_lang} to {target_lang}")
        
        # Determinar orden de proveedores a probar
        providers_to_try = []
        if preferred_provider and self._is_provider_available(preferred_provider, target_lang):
            providers_to_try.append(preferred_provider)
        
        # Agregar otros proveedores en orden de prioridad
        for provider in self.provider_priority:
            if provider not in providers_to_try and self._is_provider_available(provider, target_lang):
                providers_to_try.append(provider)
        
        # Intentar traducción con cada proveedor
        for provider in providers_to_try:
            try:
                logging.info(f"Trying translation with {provider.value}")
                result = await self.providers[provider](text, source_lang, target_lang)
                
                if result and result.strip() and result.strip() != text.strip():
                    logging.info(f"Translation successful with {provider.value}")
                    return result
                else:
                    logging.warning(f"Translation with {provider.value} returned empty or unchanged result")
                    
            except Exception as e:
                logging.error(f"Translation failed with {provider.value}: {e}")
                continue
        
        logging.error("All translation providers failed")
        return None
    
    def _is_provider_available(self, provider: TranslationProvider, target_lang: str) -> bool:
        """Verifica si un proveedor está disponible y soporta el idioma"""
        if provider == TranslationProvider.DEEPL:
            return (self.deepl_api_key and 
                   target_lang in self.language_mappings.get(provider, {}))
        elif provider == TranslationProvider.MICROSOFT:
            return (self.microsoft_api_key and 
                   target_lang in self.language_mappings.get(provider, {}))
        elif provider == TranslationProvider.LIBRETRANSLATE:
            return True  # LibreTranslate es self-hosted, asumimos disponible
        elif provider == TranslationProvider.GOOGLE:
            return True  # Google Translate como fallback
        return False
    
    async def _translate_deepl(self, text: str, source_lang: str, target_lang: str) -> Optional[str]:
        """Traducción usando DeepL API"""
        if not self.deepl_api_key:
            raise Exception("DeepL API key not configured")
        
        # Mapear códigos de idioma para DeepL
        target_lang_code = self.language_mappings[TranslationProvider.DEEPL].get(target_lang, target_lang.upper())
        
        url = "https://api-free.deepl.com/v2/translate"
        headers = {"Authorization": f"DeepL-Auth-Key {self.deepl_api_key}"}
        
        data = {
            "text": text,
            "target_lang": target_lang_code,
            "source_lang": source_lang.upper() if source_lang != "auto" else None
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, data=data) as response:
                if response.status == 200:
                    result = await response.json()
                    translations = result.get("translations", [])
                    if translations:
                        return translations[0].get("text")
                else:
                    raise Exception(f"DeepL API error: {response.status}")
        
        return None
    
    async def _translate_microsoft(self, text: str, source_lang: str, target_lang: str) -> Optional[str]:
        """Traducción usando Microsoft Translator"""
        if not self.microsoft_api_key:
            raise Exception("Microsoft Translator API key not configured")
        
        # Mapear códigos de idioma para Microsoft
        target_lang_code = self.language_mappings[TranslationProvider.MICROSOFT].get(target_lang, target_lang)
        
        endpoint = "https://api.cognitive.microsofttranslator.com"
        path = '/translate'
        constructed_url = endpoint + path
        
        params = {
            'api-version': '3.0',
            'to': target_lang_code
        }
        
        if source_lang != "auto":
            params['from'] = source_lang
        
        headers = {
            'Ocp-Apim-Subscription-Key': self.microsoft_api_key,
            'Ocp-Apim-Subscription-Region': self.microsoft_region,
            'Content-Type': 'application/json'
        }
        
        body = [{'text': text}]
        
        async with aiohttp.ClientSession() as session:
            async with session.post(constructed_url, params=params, headers=headers, json=body) as response:
                if response.status == 200:
                    result = await response.json()
                    if result and len(result) > 0:
                        translations = result[0].get("translations", [])
                        if translations:
                            return translations[0].get("text")
                else:
                    raise Exception(f"Microsoft Translator API error: {response.status}")
        
        return None
    
    async def _translate_libretranslate(self, text: str, source_lang: str, target_lang: str) -> Optional[str]:
        """Traducción usando LibreTranslate self-hosted"""
        url = f"{self.libretranslate_url}/translate"
        
        data = {
            "q": text,
            "source": source_lang if source_lang != "auto" else "auto",
            "target": target_lang,
            "format": "text"
        }
        
        headers = {"Content-Type": "application/json"}
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.post(url, headers=headers, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("translatedText")
                    else:
                        raise Exception(f"LibreTranslate API error: {response.status}")
        except asyncio.TimeoutError:
            raise Exception("LibreTranslate timeout - service may not be running")
        except aiohttp.ClientConnectorError:
            raise Exception("LibreTranslate connection failed - service may not be running")
        
        return None
    
    async def _translate_google(self, text: str, source_lang: str, target_lang: str) -> Optional[str]:
        """Traducción usando Google Translate (fallback)"""
        try:
            from deep_translator import GoogleTranslator
            
            translator = GoogleTranslator(
                source=source_lang if source_lang != "auto" else "auto",
                target=target_lang
            )
            
            # GoogleTranslator no es async, ejecutar en thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, translator.translate, text)
            return result
            
        except Exception as e:
            raise Exception(f"Google Translate error: {e}")
    
    async def translate_segments(self, 
                               segments: List[Dict], 
                               source_lang: str, 
                               target_lang: str,
                               preferred_provider: Optional[TranslationProvider] = None) -> List[Dict]:
        """
        Traduce una lista de segmentos manteniendo la estructura
        
        Args:
            segments: Lista de segmentos con estructura Whisper
            source_lang: Idioma de origen
            target_lang: Idioma destino
            preferred_provider: Proveedor preferido
            
        Returns:
            Lista de segmentos traducidos
        """
        translated_segments = []
        
        for segment in segments:
            if 'text' in segment and segment['text'].strip():
                try:
                    translated_text = await self.translate_text(
                        segment['text'], 
                        source_lang, 
                        target_lang,
                        preferred_provider
                    )
                    
                    # Crear nuevo segmento con traducción
                    new_segment = segment.copy()
                    new_segment['text'] = translated_text if translated_text else segment['text']
                    translated_segments.append(new_segment)
                    
                except Exception as e:
                    logging.error(f"Error translating segment: {e}")
                    # Mantener segmento original si falla la traducción
                    translated_segments.append(segment)
            else:
                # Mantener segmento sin texto
                translated_segments.append(segment)
        
        return translated_segments
    
    def get_supported_languages(self) -> Dict[str, List[str]]:
        """Retorna idiomas soportados por cada proveedor"""
        return {
            "deepl": ["EN", "ES", "FR", "DE", "IT", "PT", "RU", "ZH", "JA", "KO"],
            "microsoft": ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko", "ar", "hi"],
            "libretranslate": ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko", "ar"],
            "google": ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko", "ar", "hi"]
        }
    
    def get_provider_status(self) -> Dict[str, Dict[str, Any]]:
        """Retorna el estado de cada proveedor"""
        status = {}
        
        for provider in TranslationProvider:
            status[provider.value] = {
                "available": False,
                "configuration": "missing",
                "supported_languages": []
            }
            
            if provider == TranslationProvider.DEEPL:
                status[provider.value]["available"] = bool(self.deepl_api_key)
                status[provider.value]["configuration"] = "configured" if self.deepl_api_key else "missing DEEPL_API_KEY"
                status[provider.value]["supported_languages"] = list(self.language_mappings[provider].keys())
                
            elif provider == TranslationProvider.MICROSOFT:
                status[provider.value]["available"] = bool(self.microsoft_api_key)
                status[provider.value]["configuration"] = "configured" if self.microsoft_api_key else "missing MICROSOFT_TRANSLATOR_KEY"
                status[provider.value]["supported_languages"] = list(self.language_mappings[provider].keys())
                
            elif provider == TranslationProvider.LIBRETRANSLATE:
                status[provider.value]["available"] = True
                status[provider.value]["configuration"] = f"URL: {self.libretranslate_url}"
                status[provider.value]["supported_languages"] = ["en", "es", "fr", "de", "it", "pt", "ru"]
                
            elif provider == TranslationProvider.GOOGLE:
                status[provider.value]["available"] = True
                status[provider.value]["configuration"] = "fallback service"
                status[provider.value]["supported_languages"] = ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko", "ar", "hi"]
        
        return status