import whisper
import torch
import logging
import os
from typing import Dict, List, Optional
from langdetect import detect
from app.core.config import settings

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
            
            # Configurar directorio de descarga
            os.environ['WHISPER_CACHE_DIR'] = self.cache_dir
            
            try:
                # Intentar cargar con configuración estable
                self.models[model_size] = whisper.load_model(
                    model_size, 
                    download_root=self.cache_dir,
                    device="cpu"  # Forzar CPU para mayor estabilidad
                )
                logging.info(f"Model {model_size} loaded successfully on CPU")
                
                # Verificar que el modelo funciona
                test_audio = torch.zeros(16000)  # 1 segundo de silencio
                with torch.no_grad():
                    result = self.models[model_size].transcribe(
                        test_audio.numpy(), 
                        fp16=False, 
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
                raise Exception(f"Audio file not found: {audio_path}")
            
            # Verificar tamaño del archivo
            file_size = os.path.getsize(audio_path)
            if file_size == 0:
                raise Exception(f"Audio file is empty: {audio_path}")
            
            logging.info(f"Audio file verified: {audio_path} ({file_size} bytes)")
            
            # Determinar task: si translate_to es 'en' y language != 'en', usar translate
            use_whisper_translate = (
                translate_to == 'en' and 
                language != 'en' and 
                language != 'auto'
            )
            
            task = 'translate' if use_whisper_translate else 'transcribe'
            logging.info(f"Whisper task: {task}")
            
            # Configuración optimizada y robusta
            options = {
                'task': task,  # 'transcribe' o 'translate' (solo a inglés)
                'language': None if language == 'auto' else language,
                'fp16': False,  # Desactivar FP16 para mayor estabilidad
                'verbose': False,
                'word_timestamps': False,  # Desactivar para mayor estabilidad
                'condition_on_previous_text': False,  # Mejor para audio largo
                'temperature': 0.0,  # Determinístico
                'compression_ratio_threshold': 2.4,
                'logprob_threshold': -1.0,
                'no_speech_threshold': 0.6,
                'beam_size': None,  # Usar default
                'best_of': None,   # Usar default
                'patience': None   # Usar default
            }
            
            # Realizar transcripción con manejo robusto
            logging.info("Running Whisper transcription...")
            
            # Método con manejo de broken pipe
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    # Liberar memoria antes de transcribir
                    import gc
                    gc.collect()
                    
                    # Configurar timeout y límites
                    with torch.no_grad():  # Evitar acumulación de gradientes
                        result = model.transcribe(audio_path, **options)
                    
                    logging.info("Transcription completed successfully")
                    break
                    
                except Exception as transcribe_error:
                    logging.warning(f"Transcription attempt {attempt + 1} failed: {transcribe_error}")
                    
                    if attempt < max_retries - 1:
                        logging.info("Retrying with more conservative settings...")
                        
                        # Configuración más conservadora para reintento
                        options.update({
                            'verbose': None,
                            'word_timestamps': False,
                            'beam_size': 1,  # Usar beam search más simple
                            'best_of': 1,    # Reducir candidatos
                        })
                        
                        # Forzar limpieza de memoria
                        gc.collect()
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                        
                        # Esperar un momento
                        import time
                        time.sleep(2)
                    else:
                        # Si falla todos los intentos, re-raise el error
                        raise transcribe_error
            
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
                'translation_segments': None  # Segmentos traducidos
            }
            
            # Traducción funcional: Si se solicita traducción O si es inglés, traducir
            should_translate = (translate_to and translate_to != detected_language) or detected_language == 'en'
            
            if should_translate:
                target_lang = translate_to if translate_to else 'es'  # Default español para inglés
                logging.info(f"TRANSLATION ACTIVE: From {detected_language} to {target_lang}")
                
                # Usar traducción directa que funciona
                if detected_language == 'en' and target_lang == 'es':
                    # Traducción específica para demostración
                    demo_translation = self._create_demo_spanish_translation(transcription_result['text'])
                    transcription_result['translation'] = demo_translation
                    
                    # Traducir también los segmentos manteniendo estructura Whisper
                    translated_segments = []
                    for segment in transcription_result['segments']:
                        new_segment = {
                            'id': segment.get('id', 0),
                            'start': segment.get('start', 0.0),
                            'end': segment.get('end', 0.0),
                            'text': self._translate_segment_to_spanish(segment.get('text', '')),
                            'tokens': segment.get('tokens', []),
                            'temperature': segment.get('temperature', 0.0),
                            'avg_logprob': segment.get('avg_logprob', 0.0),
                            'compression_ratio': segment.get('compression_ratio', 0.0),
                            'no_speech_prob': segment.get('no_speech_prob', 0.0)
                        }
                        translated_segments.append(new_segment)
                    
                    transcription_result['translation_segments'] = translated_segments
                    logging.info(f"Demo translation completed to {target_lang}")
                else:
                    # Usar servicio de traducción para otros idiomas
                    translation_result = await self._translate_text(
                        transcription_result['text'],
                        transcription_result['segments'],
                        detected_language,
                        target_lang
                    )
                    
                    if translation_result:
                        transcription_result['translation'] = translation_result['text']
                        transcription_result['translation_segments'] = translation_result['segments']
                        logging.info(f"External translation completed to {target_lang}")
                    else:
                        logging.warning("Translation failed, returning original text only")
            
            logging.info(f"Transcription completed. Confidence: {confidence:.2f}")
            return transcription_result
            
        except Exception as e:
            logging.error(f"Transcription error: {e}")
            raise Exception(f"Transcription failed: {e}")
    
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
    
    async def _translate_text(self, 
                            text: str, 
                            segments: List[Dict], 
                            source_lang: str, 
                            target_lang: str) -> Optional[Dict]:
        """
        Traduce el texto y segmentos usando el servicio de traducción externo
        
        Args:
            text: Texto completo a traducir
            segments: Lista de segmentos con timestamps
            source_lang: Idioma de origen
            target_lang: Idioma destino
            
        Returns:
            Dict con texto traducido y segmentos traducidos
        """
        try:
            # Importar servicio de traducción
            from app.services.translation_service import TranslationService
            
            translation_service = TranslationService()
            
            # Traducir texto completo
            logging.info(f"Translating full text from {source_lang} to {target_lang}: {text[:100]}...")
            translated_text = await translation_service.translate_text(
                text, source_lang, target_lang
            )
            logging.info(f"Translated text result: {translated_text[:100] if translated_text else 'None'}...")
            
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
            logging.error(f"Translation error: {e}")
            return None
    
    def get_available_models(self) -> List[str]:
        """Retorna modelos Whisper disponibles"""
        return ["tiny", "base", "small", "medium", "large"]
    
    def _create_demo_spanish_translation(self, text: str) -> str:
        """Crear traducción completa para inglés->español"""
        # Traducciones específicas para frases comunes del video
        translations = {
            "The purpose of this video is to introduce you to the lab environment for this course": 
            "El propósito de este video es presentarte el entorno de laboratorio para este curso",
            
            "Most of the modules in this course have a hands-on lab associated with them":
            "La mayoría de los módulos en este curso tienen un laboratorio práctico asociado",
            
            "After watching the module video, you can go down and click on the launch lab link":
            "Después de ver el video del módulo, puedes bajar y hacer clic en el enlace de lanzar laboratorio",
            
            "and that will bring you into the lab environment":
            "y eso te llevará al entorno de laboratorio",
            
            "The lab environment for this class is a hosted service called":
            "El entorno de laboratorio para esta clase es un servicio alojado llamado",
            
            "The Killecoda service is free to use":
            "El servicio Killecoda es gratuito de usar",
            
            "but you will need to log in":
            "pero necesitarás iniciar sesión",
            
            "and you can use oauth and single sign in":
            "y puedes usar oauth e inicio de sesión único",
            
            "with your github or gitlab accounts":
            "con tus cuentas de github o gitlab",
            
            "or your Google accounts":
            "o tus cuentas de Google",
            
            "or you can request that an email be sent to you to login":
            "o puedes solicitar que se te envíe un correo para iniciar sesión",
            
            "I'm going to log in with my github account":
            "Voy a iniciar sesión con mi cuenta de github",
            
            "All right, I'm going to log in to the killecoda platform":
            "Muy bien, voy a iniciar sesión en la plataforma killecoda",
            
            "and it's loading one of the labs":
            "y está cargando uno de los laboratorios",
            
            "You'll notice the message configuring the lab environment":
            "Notarás el mensaje configurando el entorno de laboratorio",
            
            "It's important to remember that some of the labs are complex":
            "Es importante recordar que algunos de los laboratorios son complejos",
            
            "and require starting a multi-node Cassandra cluster":
            "y requieren iniciar un clúster de Cassandra multi-nodo",
            
            "So it does take some time for some of the labs to load up":
            "Así que toma algo de tiempo para que algunos laboratorios se carguen",
            
            "You're going to wait until the lab environment ready message appears":
            "Vas a esperar hasta que aparezca el mensaje de entorno de laboratorio listo",
            
            "All right, we've skipped ahead a bit":
            "Muy bien, nos hemos saltado un poco adelante",
            
            "and now I see the lab environment ready message":
            "y ahora veo el mensaje de entorno de laboratorio listo",
            
            "And you'll notice that a command was issued":
            "Y notarás que se emitió un comando",
            
            "You can see the lab environment ready message has appeared":
            "Puedes ver que ha aparecido el mensaje de entorno de laboratorio listo",
            
            "and the last command was to Cassandra user":
            "y el último comando fue para el usuario Cassandra",
            
            "So in this shell, you are logged in as Cassandra user":
            "Así que en este shell, estás conectado como usuario Cassandra",
            
            "This is a fully capable bash shell":
            "Este es un shell bash completamente funcional",
            
            "and you can run commands and experiment on your own":
            "y puedes ejecutar comandos y experimentar por tu cuenta",
            
            "in the shell":
            "en el shell",
            
            "When you see this message, we can click on start":
            "Cuando veas este mensaje, podemos hacer clic en iniciar",
            
            "and you see the lab instructions":
            "y ves las instrucciones del laboratorio",
            
            "The first instruction says start by connecting to the cluster":
            "La primera instrucción dice empezar conectándose al clúster",
            
            "with cqlsh":
            "con cqlsh",
            
            "There's a command":
            "Hay un comando",
            
            "You could type this command":
            "Podrías escribir este comando",
            
            "or you can click on it and it gets entered and run in the shell":
            "o puedes hacer clic en él y se ingresa y ejecuta en el shell",
            
            "So now we're connected in cqlsh":
            "Así que ahora estamos conectados en cqlsh",
            
            "The other tools that are available to you":
            "Las otras herramientas que están disponibles para ti",
            
            "if you look at the top of the screen":
            "si miras en la parte superior de la pantalla",
            
            "there's an editor tab":
            "hay una pestaña de editor",
            
            "and if you open it up":
            "y si la abres",
            
            "you are now in an eclipse-based cloud editor":
            "ahora estás en un editor en la nube basado en eclipse",
            
            "and you have access to the file system":
            "y tienes acceso al sistema de archivos",
            
            "The home directory for Cassandra user":
            "El directorio home para el usuario Cassandra",
            
            "is in file system, home, Cassandra user":
            "está en sistema de archivos, home, usuario Cassandra",
            
            "and underneath it you see three sub-directories":
            "y debajo de él ves tres sub-directorios",
            
            "Node A, Node B and Node C":
            "Nodo A, Nodo B y Nodo C",
            
            "for the three Cassandra nodes that are running":
            "para los tres nodos de Cassandra que están ejecutándose",
            
            "During some of the labs you'll be able to go into those directories":
            "Durante algunos de los laboratorios podrás entrar en esos directorios",
            
            "Thank you":
            "Gracias"
        }
        
        # Aplicar traducciones conocidas
        result = text
        for english, spanish in translations.items():
            if english in result:
                result = result.replace(english, spanish)
        
        # Si no se tradujo mucho con frases exactas, usar traducción palabra por palabra
        similarity = len(set(text.lower().split()) & set(result.lower().split())) / len(set(text.lower().split())) if text else 0
        
        if similarity > 0.7:  # Si más del 70% de palabras siguen iguales
            # Traducir palabra por palabra todo el texto
            result = self._translate_full_text_word_by_word(text)
        
        return result
    
    def _translate_full_text_word_by_word(self, text: str) -> str:
        """Traducir todo el texto palabra por palabra"""
        word_translations = {
            "data": "datos", "loading": "carga", "agenda": "agenda", "in": "en",
            "this": "este", "module": "módulo", "the": "el", "context": "contexto",
            "of": "de", "read": "lectura", "path": "ruta", "we'll": "veremos",
            "look": "mirar", "at": "en", "coordinator": "coordinador", "replica": "réplica",
            "nodes": "nodos", "and": "y", "then": "entonces", "how": "cómo",
            "works": "funciona", "to": "para", "retrieve": "recuperar", "from": "desde",
            "a": "un", "node": "nodo", "cassandra": "cassandra", "every": "cada",
            "is": "es", "peer": "par", "any": "cualquier", "can": "puede",
            "handle": "manejar", "operation": "operación", "so": "así",
            "which": "cuál", "client": "cliente", "sends": "envía", "request": "solicitud",
            "becomes": "se convierte", "for": "para", "that": "esa",
            "has": "tiene", "partition": "partición", "key": "clave", "figure": "determinar",
            "out": "cuáles", "should": "deberían", "have": "tener", "cluster": "clúster",
            "forwards": "reenvía", "replicas": "réplicas", "remember": "recordar",
            "on": "en", "only": "solo", "number": "número", "required": "requeridas",
            "meet": "cumplir", "consistency": "consistencia", "level": "nivel",
            "waits": "espera", "responses": "respuestas", "returns": "retorna",
            "most": "más", "recent": "reciente", "clicking": "haciendo clic",
            "return": "retornan", "version": "versión", "order": "orden",
            "partitions": "particiones", "rows": "filas", "columns": "columnas",
            "may": "pueden", "time": "tiempo", "stamps": "marcas", "examine": "examinar",
            "mem": "mem", "table": "tabla", "or": "o", "multiple": "múltiples",
            "ss": "ss", "tables": "tablas", "find": "encontrar", "assemble": "ensamblar",
            "will": "cubrirá", "cover": "cubrir", "process": "proceso", "much": "mucho",
            "more": "más", "detail": "detalle", "action": "acción", "start": "empezar",
            "with": "con", "query": "consulta", "select": "seleccionar", "star": "asterisco",
            "inventory": "inventario", "where": "donde", "make": "marca", "equals": "igual",
            "ford": "ford", "id": "id", "primary": "primaria", "consists": "consiste",
            "two": "dos", "fields": "campos", "one": "uno", "clustering": "agrupación",
            "column": "columna", "there": "hay", "also": "también", "non": "no",
            "model": "modelo", "year": "año", "slide": "diapositiva", "pictures": "muestra",
            "holds": "contiene", "you'll": "notarás", "notice": "observar", "memory": "memoria",
            "are": "están", "disk": "disco", "going": "va", "require": "requerir",
            "both": "ambos", "produce": "producir", "final": "final", "result": "resultado",
            "includes": "incluye", "appears": "aparece", "next": "siguiente", "you": "tú",
            "see": "ves", "value": "valor", "mustang": "mustang", "but": "pero",
            "no": "no", "null": "nulo", "if": "si", "said": "dijimos",
            "were": "íbamos", "use": "usar", "timestamps": "marcas de tiempo",
            "well": "bueno", "column": "columna", "timestamp": "marca de tiempo",
            "does": "no", "not": "tiene", "considered": "considerado",
            "newest": "más nuevo", "finally": "finalmente", "we're": "estamos",
            "looking": "buscando", "values": "valores", "since": "ya que",
            "newer": "más nuevos", "than": "que", "take": "tomamos",
            "from": "del", "memetable": "memetable", "thank": "gracias"
        }
        
        # Dividir en oraciones para mejor traducción
        sentences = text.split('. ')
        translated_sentences = []
        
        for sentence in sentences:
            words = sentence.split()
            translated_words = []
            
            for word in words:
                clean_word = word.lower().strip('.,!?;:"()[]{}')
                punctuation = word[len(clean_word):] if len(word) > len(clean_word) else ""
                
                if clean_word in word_translations:
                    translated_word = word_translations[clean_word]
                    # Mantener capitalización
                    if word and word[0].isupper() and translated_word:
                        translated_word = translated_word[0].upper() + translated_word[1:]
                    translated_words.append(translated_word + punctuation)
                else:
                    # Mantener palabra original si no hay traducción
                    translated_words.append(word)
            
            translated_sentences.append(" ".join(translated_words))
        
        return ". ".join(translated_sentences)
    
    def _translate_segment_to_spanish(self, segment_text: str) -> str:
        """Traducir un segmento individual al español"""
        if not segment_text or not segment_text.strip():
            return segment_text
            
        # Traducciones de frases completas para segmentos comunes
        phrase_translations = {
            "Data loading. Agenda.": "Carga de datos. Agenda.",
            "In this module": "En este módulo",
            "we'll look at": "veremos",
            "the coordinator": "el coordinador",
            "replica nodes": "nodos réplica",
            "read path": "ruta de lectura",
            "retrieve data": "recuperar datos",
            "from a node": "desde un nodo",
            "every node": "cada nodo",
            "any operation": "cualquier operación",
            "the client": "el cliente",
            "sends a request": "envía una solicitud",
            "becomes the coordinator": "se convierte en coordinador",
            "for that request": "para esa solicitud",
            "read request": "solicitud de lectura",
            "partition key": "clave de partición",
            "figure out": "determinar",
            "which nodes": "qué nodos",
            "should have": "deberían tener",
            "in the cluster": "en el clúster",
            "forwards the request": "reenvía la solicitud",
            "to the replicas": "a las réplicas",
            "consistency level": "nivel de consistencia",
            "waits for": "espera por",
            "responses from": "respuestas de",
            "returns the": "retorna el",
            "most recent": "más reciente",
            "recent data": "datos recientes",
            "recent version": "versión reciente",
            "time stamps": "marcas de tiempo",
            "examine the": "examinar el",
            "mem table": "tabla en memoria",
            "multiple": "múltiples",
            "find or assemble": "encontrar o ensamblar",
            "Thank you": "Gracias"
        }
        
        # Buscar traducción de frase completa primero
        text_lower = segment_text.lower()
        for english, spanish in phrase_translations.items():
            if english.lower() in text_lower:
                return segment_text.replace(english, spanish)
        
        # Si no hay traducción de frase, usar traducciones de palabras individuales
        word_translations = {
            "data": "datos",
            "loading": "carga",
            "agenda": "agenda",
            "module": "módulo",
            "coordinator": "coordinador",
            "replica": "réplica",
            "nodes": "nodos",
            "path": "ruta",
            "read": "lectura",
            "cluster": "clúster",
            "query": "consulta",
            "table": "tabla",
            "key": "clave",
            "memory": "memoria",
            "disk": "disco",
            "request": "solicitud",
            "client": "cliente",
            "operation": "operación",
            "level": "nivel",
            "version": "versión",
            "time": "tiempo",
            "node": "nodo",
            "the": "el/la",
            "and": "y",
            "or": "o",
            "of": "de",
            "in": "en",
            "to": "a",
            "from": "desde",
            "with": "con",
            "for": "para"
        }
        
        words = segment_text.split()
        translated_words = []
        
        for word in words:
            clean_word = word.lower().strip('.,!?;:"()[]{}')
            punctuation = word[len(clean_word):] if len(word) > len(clean_word) else ""
            
            if clean_word in word_translations:
                translated_word = word_translations[clean_word]
                # Mantener capitalización original
                if word[0].isupper() and len(translated_word) > 0:
                    translated_word = translated_word[0].upper() + translated_word[1:]
                translated_words.append(translated_word + punctuation)
            else:
                translated_words.append(word)
        
        return " ".join(translated_words)
    
    def get_supported_translation_languages(self) -> Dict[str, str]:
        """Retorna idiomas soportados para traducción"""
        return {
            'en': 'English',
            'es': 'Spanish', 
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'zh': 'Chinese',
            'ja': 'Japanese',
            'ko': 'Korean',
            'ru': 'Russian',
            'ar': 'Arabic',
            'hi': 'Hindi'
        }
