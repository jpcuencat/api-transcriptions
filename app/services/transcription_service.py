import whisper
import torch
import logging
import os
import re
from typing import Dict, List, Optional
from langdetect import detect
from app.core.config import settings
import gc
import time
from transformers import MarianMTModel, MarianTokenizer, AutoTokenizer, AutoModelForSeq2SeqLM
from jiwer import wer

class TranscriptionService:
    def __init__(self):
        self.models = {}
        self.cache_dir = settings.WHISPER_CACHE_DIR

        # Crear directorio de cache si no existe
        os.makedirs(self.cache_dir, exist_ok=True)

    def load_model(self, model_size: str = "base") -> whisper.Whisper:
        """Carga modelo Whisper con cache y manejo robusto de errores"""
        if model_size not in self.models:
            logging.info(f"Loading Whisper model: {model_size}")

            # Detectar GPU automáticamente
            device = "cuda" if torch.cuda.is_available() else "cpu"
            os.environ['WHISPER_CACHE_DIR'] = self.cache_dir

            try:
                # Intentar cargar con configuración estable
                self.models[model_size] = whisper.load_model(
                    model_size, 
                    download_root=self.cache_dir,
                    device=device
                )
                logging.info(f"Model {model_size} loaded successfully on {device}")

                # Verificar que el modelo funciona
                test_audio = torch.zeros(16000)  # 1 segundo de silencio
                with torch.no_grad():
                    result = self.models[model_size].transcribe(
                        test_audio.numpy(), 
                        fp16=torch.cuda.is_available(), 
                        verbose=False,
                        word_timestamps=False
                    )
                logging.info("Model verification successful")

            except Exception as e:
                logging.error(f"Error loading model {model_size}: {e}")

                # Intentar limpieza y recarga
                if model_size in self.models:
                    del self.models[model_size]

                # Reintento con configuración minimal
                try:
                    logging.info("Retrying model load with minimal configuration...")
                    self.models[model_size] = whisper.load_model(model_size)
                    logging.info(f"Model {model_size} loaded on retry")
                except Exception as retry_error:
                    logging.error(f"Model reload failed: {retry_error}")
                    raise Exception(f"Failed to load Whisper model after retry: {retry_error}")

        return self.models[model_size]

    def evaluate_transcription(self, reference: str, hypothesis: str) -> float:
        """Evalúa la calidad de la transcripción usando WER"""
        return wer(reference, hypothesis)

    def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        """Traduce texto usando modelos locales descargados"""
        try:
            logging.info(f"Solicitada traducción de {source_lang} a {target_lang}")
            
            # Normalizar códigos de idioma a 2 letras
            source_normalized = source_lang.lower()[:2] if source_lang else 'en'
            target_normalized = target_lang.lower()[:2] if target_lang else 'es'
            
            logging.info(f"Traducción normalizada: {source_normalized} -> {target_normalized}")
            
            # Si el idioma origen y destino son iguales, no traducir
            if source_normalized == target_normalized:
                logging.info("Idiomas origen y destino son iguales, retornando texto original")
                return text
            
            # Usar modelo local para traducción
            return self._translate_with_local_model(text, source_normalized, target_normalized)
                
        except Exception as e:
            logging.error(f"Error en la traducción: {e}")
            return text

    def _translate_with_local_model(self, text: str, source_lang: str, target_lang: str) -> str:
        """Traduce usando modelos locales descargados almacenados en el proyecto"""
        try:
            # Ruta local para modelos de traducción
            models_dir = os.path.join(settings.BASE_DIR if hasattr(settings, 'BASE_DIR') else os.getcwd(), "models", "translation")
            
            # Crear directorio si no existe
            os.makedirs(models_dir, exist_ok=True)
            
            # Mapeo completo de modelos según idiomas (ampliado)
            model_mapping = {
                # Inglés <-> Otros idiomas
                ('en', 'es'): 'Helsinki-NLP/opus-mt-en-es',
                ('es', 'en'): 'Helsinki-NLP/opus-mt-es-en', 
                ('en', 'fr'): 'Helsinki-NLP/opus-mt-en-fr',
                ('fr', 'en'): 'Helsinki-NLP/opus-mt-fr-en',
                ('en', 'de'): 'Helsinki-NLP/opus-mt-en-de',
                ('de', 'en'): 'Helsinki-NLP/opus-mt-de-en',
                ('en', 'it'): 'Helsinki-NLP/opus-mt-en-it',
                ('it', 'en'): 'Helsinki-NLP/opus-mt-it-en',
                ('en', 'pt'): 'Helsinki-NLP/opus-mt-en-pt',
                ('pt', 'en'): 'Helsinki-NLP/opus-mt-pt-en',
                ('en', 'ru'): 'Helsinki-NLP/opus-mt-en-ru',
                ('ru', 'en'): 'Helsinki-NLP/opus-mt-ru-en',
                
                # Español <-> Otros idiomas
                ('es', 'pt'): 'Helsinki-NLP/opus-mt-es-pt',
                ('pt', 'es'): 'Helsinki-NLP/opus-mt-pt-es',
                ('es', 'fr'): 'Helsinki-NLP/opus-mt-es-fr',
                ('fr', 'es'): 'Helsinki-NLP/opus-mt-fr-es',
                ('es', 'de'): 'Helsinki-NLP/opus-mt-es-de',
                ('de', 'es'): 'Helsinki-NLP/opus-mt-de-es',
                ('es', 'it'): 'Helsinki-NLP/opus-mt-es-it',
                ('it', 'es'): 'Helsinki-NLP/opus-mt-it-es',
                
                # Francés <-> Otros idiomas (además de inglés/español)
                ('fr', 'de'): 'Helsinki-NLP/opus-mt-fr-de',
                ('de', 'fr'): 'Helsinki-NLP/opus-mt-de-fr',
                ('fr', 'pt'): 'Helsinki-NLP/opus-mt-fr-pt',
                ('pt', 'fr'): 'Helsinki-NLP/opus-mt-pt-fr',
                
                # Otros pares comunes
                ('de', 'pt'): 'Helsinki-NLP/opus-mt-de-pt',
                ('pt', 'de'): 'Helsinki-NLP/opus-mt-pt-de',
                ('it', 'pt'): 'Helsinki-NLP/opus-mt-it-pt',
                ('pt', 'it'): 'Helsinki-NLP/opus-mt-pt-it'
            }
            
            # Normalizar códigos de idioma
            source_normalized = source_lang.lower()[:2]
            target_normalized = target_lang.lower()[:2]
            
            # Obtener modelo para el par de idiomas
            model_key = (source_normalized, target_normalized)
            
            if model_key not in model_mapping:
                logging.warning(f"No hay modelo local para {source_normalized} -> {target_normalized}")
                return self._fallback_translation(text, source_lang, target_lang)
            
            model_name = model_mapping[model_key]
            model_path = os.path.join(models_dir, model_name.replace('/', '_'))
            
            # Intentar cargar modelo local primero
            if os.path.exists(model_path):
                logging.info(f"Cargando modelo local desde: {model_path}")
                tokenizer = MarianTokenizer.from_pretrained(model_path)
                model = MarianMTModel.from_pretrained(model_path)
            else:
                # Descargar y guardar modelo localmente
                logging.info(f"Descargando modelo {model_name} a {model_path}")
                
                try:
                    tokenizer = MarianTokenizer.from_pretrained(model_name)
                    model = MarianMTModel.from_pretrained(model_name)
                    
                    # Guardar modelo localmente para uso futuro
                    tokenizer.save_pretrained(model_path)
                    model.save_pretrained(model_path)
                    
                    logging.info(f"Modelo guardado localmente en: {model_path}")
                    
                except Exception as download_error:
                    logging.error(f"Error descargando modelo {model_name}: {download_error}")
                    return self._fallback_translation(text, source_lang, target_lang)
            
            # Realizar traducción
            logging.info(f"Traduciendo con modelo local: {text[:100]}...")
            
            # Dividir texto en chunks para manejar textos largos
            chunks = self._split_text_for_translation(text)
            translated_chunks = []
            
            for chunk in chunks:
                if chunk.strip():
                    # Tokenizar
                    inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=512)
                    
                    # Traducir
                    with torch.no_grad():
                        translated = model.generate(**inputs, max_length=512, num_beams=4, early_stopping=True)
                    
                    # Decodificar
                    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
                    translated_chunks.append(translated_text)
                else:
                    translated_chunks.append(chunk)
            
            result = ' '.join(translated_chunks)
            logging.info(f"Traducción completada: {result[:100]}...")
            return result
            
        except Exception as e:
            logging.error(f"Error en traducción con modelo local: {e}")
            return self._fallback_translation(text, source_lang, target_lang)
    
    def _split_text_for_translation(self, text: str, max_length: int = 400) -> List[str]:
        """Divide texto en chunks manejables para traducción"""
        if len(text) <= max_length:
            return [text]
        
        # Dividir por oraciones primero
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + " " + sentence) <= max_length:
                current_chunk = current_chunk + " " + sentence if current_chunk else sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _fallback_translation(self, text: str, source_lang: str, target_lang: str) -> str:
        """Método de fallback cuando no se puede usar modelo local"""
        logging.info(f"Usando fallback para traducción {source_lang} -> {target_lang}")
        
        if target_lang.lower().startswith('es'):
            return self._create_simple_spanish_translation(text)
        elif target_lang.lower().startswith('fr'):
            return self._create_simple_french_translation(text)
        elif target_lang.lower().startswith('de'):
            return self._create_simple_german_translation(text)
        elif target_lang.lower().startswith('it'):
            return self._create_simple_italian_translation(text)
        elif target_lang.lower().startswith('pt'):
            return self._create_simple_portuguese_translation(text)
        else:
            logging.warning(f"No hay fallback para {target_lang}")
            return text

    def _create_simple_spanish_translation(self, text: str) -> str:
        """Traducción simple palabra por palabra al español"""
        basic_dict = {
            # Palabras básicas
            "hello": "hola", "hi": "hola", "the": "el", "a": "un", "and": "y",
            "or": "o", "but": "pero", "if": "si", "when": "cuando", "where": "donde",
            "what": "qué", "how": "cómo", "why": "por qué", "who": "quién",
            "this": "este", "that": "ese", "these": "estos", "those": "esos",
            "is": "es", "are": "son", "was": "era", "were": "eran", "be": "ser",
            "have": "tener", "has": "tiene", "had": "tenía", "do": "hacer",
            "does": "hace", "did": "hizo", "will": "será", "would": "haría",
            "can": "puede", "could": "podría", "should": "debería", "must": "debe",
            "may": "puede", "might": "podría", "shall": "deberá",
            "i": "yo", "you": "tú", "he": "él", "she": "ella", "it": "eso",
            "we": "nosotros", "they": "ellos", "me": "me", "him": "él", "her": "ella",
            "us": "nosotros", "them": "ellos", "my": "mi", "your": "tu", "his": "su",
            "its": "su", "our": "nuestro", "their": "su",
            "yes": "sí", "no": "no", "not": "no", "very": "muy", "so": "tan",
            "more": "más", "most": "más", "less": "menos", "much": "mucho",
            "many": "muchos", "few": "pocos", "some": "algunos", "all": "todos",
            "every": "cada", "each": "cada", "any": "cualquier", "no": "ningún",
            "good": "bueno", "bad": "malo", "big": "grande", "small": "pequeño",
            "new": "nuevo", "old": "viejo", "first": "primero", "last": "último",
            "next": "siguiente", "other": "otro", "same": "mismo", "different": "diferente",
            "work": "trabajo", "time": "tiempo", "day": "día", "year": "año",
            "way": "manera", "man": "hombre", "woman": "mujer", "person": "persona",
            "people": "gente", "child": "niño", "life": "vida", "world": "mundo",
            "hand": "mano", "part": "parte", "place": "lugar", "case": "caso",
            "right": "derecho", "thing": "cosa", "fact": "hecho", "question": "pregunta",
            "problem": "problema", "service": "servicio", "money": "dinero",
            "business": "negocio", "system": "sistema", "program": "programa",
            "number": "número", "point": "punto", "water": "agua", "room": "habitación",
            "mother": "madre", "father": "padre", "family": "familia", "friend": "amigo",
            "house": "casa", "home": "hogar", "school": "escuela", "job": "trabajo",
            "come": "venir", "go": "ir", "get": "obtener", "make": "hacer", "take": "tomar",
            "see": "ver", "know": "saber", "think": "pensar", "look": "mirar",
            "use": "usar", "find": "encontrar", "give": "dar", "tell": "decir",
            "ask": "preguntar", "try": "intentar", "need": "necesitar", "want": "querer",
            "put": "poner", "say": "decir", "call": "llamar", "turn": "girar",
            "move": "mover", "live": "vivir", "feel": "sentir", "become": "convertirse",
            "leave": "dejar", "bring": "traer", "begin": "comenzar", "keep": "mantener",
            "hold": "sostener", "write": "escribir", "sit": "sentarse", "stand": "pararse",
            "run": "correr", "walk": "caminar", "talk": "hablar", "speak": "hablar",
            "read": "leer", "play": "jugar", "open": "abrir", "close": "cerrar",
            "start": "empezar", "stop": "parar", "help": "ayudar", "change": "cambiar",
            "order": "orden", "buy": "comprar", "pay": "pagar", "meet": "conocer",
            "build": "construir", "grow": "crecer", "learn": "aprender", "teach": "enseñar",
            "today": "hoy", "tomorrow": "mañana", "yesterday": "ayer", "now": "ahora",
            "then": "entonces", "here": "aquí", "there": "allí", "where": "donde",
            "always": "siempre", "never": "nunca", "sometimes": "a veces", "often": "a menudo",
            "again": "otra vez", "also": "también", "still": "aún", "just": "solo",
            "only": "solo", "even": "incluso", "well": "bien", "back": "atrás",
            "up": "arriba", "down": "abajo", "out": "fuera", "in": "en", "on": "en",
            "at": "en", "to": "a", "for": "para", "with": "con", "by": "por",
            "from": "de", "about": "sobre", "into": "en", "through": "a través",
            "during": "durante", "before": "antes", "after": "después", "above": "encima",
            "below": "debajo", "between": "entre", "among": "entre", "around": "alrededor"
        }
        
        words = text.split()
        translated_words = []
        
        for word in words:
            clean_word = word.lower().strip('.,!?;:"()[]{}')
            punctuation = word[len(clean_word):] if len(word) > len(clean_word) else ""
            
            if clean_word in basic_dict:
                translated_word = basic_dict[clean_word]
                if word and word[0].isupper() and translated_word:
                    translated_word = translated_word[0].upper() + translated_word[1:]
                translated_words.append(translated_word + punctuation)
            else:
                translated_words.append(word)
        
        return " ".join(translated_words)
    
    def _create_simple_french_translation(self, text: str) -> str:
        """Traducción simple palabra por palabra al francés"""
        basic_dict = {
            "hello": "bonjour", "hi": "salut", "the": "le", "a": "un", "and": "et",
            "or": "ou", "but": "mais", "if": "si", "when": "quand", "where": "où",
            "what": "quoi", "how": "comment", "why": "pourquoi", "who": "qui",
            "this": "ce", "that": "cela", "is": "est", "are": "sont", "have": "avoir",
            "yes": "oui", "no": "non", "good": "bon", "bad": "mauvais", "big": "grand",
            "small": "petit", "new": "nouveau", "old": "vieux", "time": "temps",
            "work": "travail", "day": "jour", "year": "année", "people": "gens",
            "water": "eau", "house": "maison", "come": "venir", "go": "aller",
            "see": "voir", "know": "savoir", "make": "faire", "take": "prendre"
        }
        
        words = text.split()
        translated_words = []
        
        for word in words:
            clean_word = word.lower().strip('.,!?;:"()[]{}')
            punctuation = word[len(clean_word):] if len(word) > len(clean_word) else ""
            
            if clean_word in basic_dict:
                translated_word = basic_dict[clean_word]
                if word and word[0].isupper() and translated_word:
                    translated_word = translated_word[0].upper() + translated_word[1:]
                translated_words.append(translated_word + punctuation)
            else:
                translated_words.append(word)
        
        return " ".join(translated_words)
    
    def _create_simple_german_translation(self, text: str) -> str:
        """Traducción simple palabra por palabra al alemán"""
        basic_dict = {
            "hello": "hallo", "hi": "hallo", "the": "der", "a": "ein", "and": "und",
            "or": "oder", "but": "aber", "if": "wenn", "when": "wann", "where": "wo",
            "what": "was", "how": "wie", "why": "warum", "who": "wer",
            "this": "dies", "that": "das", "is": "ist", "are": "sind", "have": "haben",
            "yes": "ja", "no": "nein", "good": "gut", "bad": "schlecht", "big": "groß",
            "small": "klein", "new": "neu", "old": "alt", "time": "Zeit",
            "work": "Arbeit", "day": "Tag", "year": "Jahr", "people": "Leute",
            "water": "Wasser", "house": "Haus", "come": "kommen", "go": "gehen",
            "see": "sehen", "know": "wissen", "make": "machen", "take": "nehmen"
        }
        
        words = text.split()
        translated_words = []
        
        for word in words:
            clean_word = word.lower().strip('.,!?;:"()[]{}')
            punctuation = word[len(clean_word):] if len(word) > len(clean_word) else ""
            
            if clean_word in basic_dict:
                translated_word = basic_dict[clean_word]
                if word and word[0].isupper() and translated_word:
                    translated_word = translated_word[0].upper() + translated_word[1:]
                translated_words.append(translated_word + punctuation)
            else:
                translated_words.append(word)
        
        return " ".join(translated_words)
    
    def _create_simple_italian_translation(self, text: str) -> str:
        """Traducción simple palabra por palabra al italiano"""
        basic_dict = {
            "hello": "ciao", "hi": "ciao", "the": "il", "a": "un", "and": "e",
            "or": "o", "but": "ma", "if": "se", "when": "quando", "where": "dove",
            "what": "cosa", "how": "come", "why": "perché", "who": "chi",
            "this": "questo", "that": "quello", "is": "è", "are": "sono", "have": "avere",
            "yes": "sì", "no": "no", "good": "buono", "bad": "cattivo", "big": "grande",
            "small": "piccolo", "new": "nuovo", "old": "vecchio", "time": "tempo",
            "work": "lavoro", "day": "giorno", "year": "anno", "people": "persone",
            "water": "acqua", "house": "casa", "come": "venire", "go": "andare",
            "see": "vedere", "know": "sapere", "make": "fare", "take": "prendere"
        }
        
        words = text.split()
        translated_words = []
        
        for word in words:
            clean_word = word.lower().strip('.,!?;:"()[]{}')
            punctuation = word[len(clean_word):] if len(word) > len(clean_word) else ""
            
            if clean_word in basic_dict:
                translated_word = basic_dict[clean_word]
                if word and word[0].isupper() and translated_word:
                    translated_word = translated_word[0].upper() + translated_word[1:]
                translated_words.append(translated_word + punctuation)
            else:
                translated_words.append(word)
        
        return " ".join(translated_words)
    
    def _create_simple_portuguese_translation(self, text: str) -> str:
        """Traducción simple palabra por palabra al portugués"""
        basic_dict = {
            "hello": "olá", "hi": "oi", "the": "o", "a": "um", "and": "e",
            "or": "ou", "but": "mas", "if": "se", "when": "quando", "where": "onde",
            "what": "o que", "how": "como", "why": "por que", "who": "quem",
            "this": "este", "that": "esse", "is": "é", "are": "são", "have": "ter",
            "yes": "sim", "no": "não", "good": "bom", "bad": "mau", "big": "grande",
            "small": "pequeno", "new": "novo", "old": "velho", "time": "tempo",
            "work": "trabalho", "day": "dia", "year": "ano", "people": "pessoas",
            "water": "água", "house": "casa", "come": "vir", "go": "ir",
            "see": "ver", "know": "saber", "make": "fazer", "take": "tomar"
        }
        
        words = text.split()
        translated_words = []
        
        for word in words:
            clean_word = word.lower().strip('.,!?;:"()[]{}')
            punctuation = word[len(clean_word):] if len(word) > len(clean_word) else ""
            
            if clean_word in basic_dict:
                translated_word = basic_dict[clean_word]
                if word and word[0].isupper() and translated_word:
                    translated_word = translated_word[0].upper() + translated_word[1:]
                translated_words.append(translated_word + punctuation)
            else:
                translated_words.append(word)
        
        return " ".join(translated_words)

    def _calculate_confidence(self, result: Dict) -> float:
        """Calcula confianza promedio de la transcripción"""
        if 'segments' not in result or not result['segments']:
            return 0.0
        
        confidences = []
        for segment in result['segments']:
            # Whisper no siempre incluye word-level probabilities
            # Usar avg_logprob del segmento como aproximación
            if 'avg_logprob' in segment:
                # Convertir log probability a probability
                prob = min(1.0, max(0.0, torch.exp(torch.tensor(segment['avg_logprob'])).item()))
                confidences.append(prob)
            elif 'words' in segment:
                # Si hay word-level timestamps, usar esas probabilidades
                for word in segment['words']:
                    if 'probability' in word:
                        confidences.append(word['probability'])
        
        return sum(confidences) / len(confidences) if confidences else 0.0
    
    async def transcribe_audio(self,
                              audio_path: str,
                              language: str = 'auto',
                              model_size: str = 'base',
                              translate_to: Optional[str] = None) -> Dict:
        """
        Transcribe y opcionalmente traduce audio usando Whisper
        
        Args:
            audio_path: Ruta al archivo de audio
            language: Idioma del audio ('auto' para detección automática)
            model_size: Tamaño del modelo Whisper 
            translate_to: Idioma destino para traducción (None = solo transcripción)
            
        Returns:
            Dict con texto transcrito, idioma detectado, segmentos y traducción si aplica
        """
        try:
            logging.info(f"Starting transcription: {audio_path}")
            logging.info(f"Parameters - Language: {language}, Model: {model_size}, Translate to: {translate_to}")
            
            model = self.load_model(model_size)
            
            # Verificar que el archivo de audio existe
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            # Verificar tamaño del archivo
            file_size = os.path.getsize(audio_path)
            if file_size == 0:
                raise ValueError(f"Audio file is empty: {audio_path}")
            
            logging.info(f"Audio file verified: {audio_path} ({file_size} bytes)")
            
            # Configuración dinámica según tamaño del archivo
            options = {
                'task': 'transcribe',
                'language': None if language == 'auto' else language,
                'fp16': torch.cuda.is_available(),
                'verbose': False,
                'word_timestamps': False,
                'beam_size': 5 if file_size > 10 * 1024 * 1024 else 1,
                'best_of': 3 if file_size > 10 * 1024 * 1024 else 1
            }
            
            # Realizar transcripción con manejo robusto
            logging.info("Running Whisper transcription...")
            
            with torch.no_grad():
                result = model.transcribe(audio_path, **options)
            
            logging.info("Transcription completed successfully")
            
            # Detectar idioma si es auto
            detected_language = result.get('language', 'unknown')
            if language == 'auto':
                logging.info(f"Detected language: {detected_language}")
            
            # Calcular confidence score
            confidence = self._calculate_confidence(result)
            
            # Preparar resultado base
            transcription_result = {
                'text': result['text'].strip(),
                'language': detected_language,
                'segments': result.get('segments', []),
                'duration': result.get('duration', 0),
                'confidence': confidence,
                'translation': None,  # Se llenará si hay traducción
                'translation_segments': None  # Se llenará si hay traducción
            }
            
            # Traducción avanzada
            if translate_to and translate_to != detected_language:
                logging.info(f"Iniciando traducción de {len(transcription_result['segments'])} segmentos")
                
                # Traducir texto completo
                transcription_result['translation'] = self.translate_text(
                    transcription_result['text'], detected_language, translate_to
                )
                
                # Traducir segmentos individuales
                translated_segments = []
                for i, segment in enumerate(transcription_result['segments']):
                    if 'text' in segment and segment['text'].strip():
                        logging.debug(f"Traduciendo segmento {i}: '{segment['text'][:50]}...'")
                        translated_segment = segment.copy()
                        translated_segment['translation'] = self.translate_text(
                            segment['text'], detected_language, translate_to
                        )
                        translated_segments.append(translated_segment)
                    else:
                        # IMPORTANTE: Conservar segmentos vacíos para mantener la estructura temporal
                        logging.debug(f"Conservando segmento vacío {i} para mantener timing: {segment}")
                        empty_segment = segment.copy()
                        empty_segment['translation'] = ""  # Traducción vacía pero mantener segmento
                        translated_segments.append(empty_segment)
                
                transcription_result['translation_segments'] = translated_segments
                logging.info(f"Translation completed for {len(translated_segments)} segments")
                logging.info(f"Originales: {len(transcription_result['segments'])}, Traducidos: {len(translated_segments)}")

            logging.info(f"Transcription completed. Confidence: {confidence:.2f}")
            logging.info(f"Total segments in result: {len(transcription_result.get('segments', []))}")
            return transcription_result

        except FileNotFoundError as fnf_error:
            logging.error(f"File not found error: {fnf_error}")
            raise
        except ValueError as val_error:
            logging.error(f"Value error: {val_error}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error during transcription: {e}")
            raise Exception(f"Transcription failed: {e}")
    
    async def _translate_with_advanced_service(self, 
                                         text: str, 
                                         segments: List[Dict], 
                                         source_lang: str, 
                                         target_lang: str) -> Optional[Dict]:
        """
        Traduce el texto y segmentos usando el servicio de traducción avanzado
        
        Args:
            text: Texto completo a traducir
            segments: Lista de segmentos con timestamps
            source_lang: Idioma de origen
            target_lang: Idioma destino
            
        Returns:
            Dict con texto traducido y segmentos traducidos
        """
        try:
            # Importar servicio de traducción avanzado
            from app.services.advanced_translation_service import AdvancedTranslationService, TranslationProvider
            
            advanced_service = AdvancedTranslationService()
            
            # Traducir texto completo con el mejor proveedor disponible
            logging.info(f"Advanced translation: {text[:100]}... from {source_lang} to {target_lang}")
            translated_text = await advanced_service.translate_text(
                text, source_lang, target_lang, preferred_provider=TranslationProvider.DEEPL
            )
            logging.info(f"Advanced translation result: {translated_text[:100] if translated_text else 'None'}...")
            
            # Traducir segmentos usando el servicio avanzado
            translated_segments = await advanced_service.translate_segments(
                segments, source_lang, target_lang, preferred_provider=TranslationProvider.DEEPL
            )
            
            return {
                'text': translated_text,
                'segments': translated_segments
            }
            
        except Exception as e:
            logging.error(f"Advanced translation error: {e}")
            # Fallback al servicio original
            return await self._translate_text_fallback(text, segments, source_lang, target_lang)
    
    async def _translate_text_fallback(self, 
                                     text: str, 
                                     segments: List[Dict], 
                                     source_lang: str, 
                                     target_lang: str) -> Optional[Dict]:
        """Método de fallback usando el servicio de traducción original"""
        try:
            # Importar servicio de traducción original
            from app.services.translation_service import TranslationService
            
            translation_service = TranslationService()
            
            # Traducir texto completo
            logging.info(f"Fallback translation: {text[:100]}... from {source_lang} to {target_lang}")
            translated_text = await translation_service.translate_text(
                text, source_lang, target_lang
            )
            logging.info(f"Fallback translation result: {translated_text[:100] if translated_text else 'None'}...")
            
            # Traducir segmentos individuales para mantener timestamps
            translated_segments = []
            for segment in segments:
                if 'text' in segment:
                    translated_segment_text = await translation_service.translate_text(
                        segment['text'], source_lang, target_lang
                    )
                    
                    # Crear nuevo segmento con traducción
                    translated_segment = segment.copy()
                    translated_segment['text'] = translated_segment_text
                    translated_segments.append(translated_segment)
                else:
                    # Mantener segmento original si no tiene texto
                    translated_segments.append(segment)
            
            return {
                'text': translated_text,
                'segments': translated_segments
            }
            
        except Exception as e:
            logging.error(f"Fallback translation error: {e}")
            return None
    
    def get_available_models(self) -> List[str]:
        """Retorna modelos Whisper disponibles"""
        return ["tiny", "base", "small", "medium", "large"]
    
    def get_supported_translation_languages(self) -> Dict[str, str]:
        """Retorna idiomas soportados para traducción basado en modelos disponibles"""
        # Solo devolver idiomas que realmente tienen modelos mapeados
        # Basado en el model_mapping definido en _translate_with_local_model
        return {
            'en': 'English',
            'es': 'Spanish', 
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'ru': 'Russian'
        }
    
    def get_available_translation_pairs(self) -> Dict[str, List[str]]:
        """Retorna pares de idiomas disponibles para traducción"""
        # Mapeo basado en los modelos que realmente tenemos disponibles
        translation_pairs = {
            'en': ['es', 'fr', 'de', 'it', 'pt', 'ru'],  # Desde inglés a otros
            'es': ['en', 'pt', 'fr', 'de', 'it'],        # Desde español a otros
            'fr': ['en', 'es', 'de', 'pt'],              # Desde francés a otros  
            'de': ['en', 'es', 'fr', 'pt'],              # Desde alemán a otros
            'it': ['en', 'es', 'pt'],                    # Desde italiano a otros
            'pt': ['en', 'es', 'fr', 'de', 'it'],        # Desde portugués a otros
            'ru': ['en']                                  # Desde ruso solo a inglés
        }
        
        return translation_pairs
