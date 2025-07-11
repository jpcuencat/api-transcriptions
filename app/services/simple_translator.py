"""
Traductor simple para demostrar funcionalidad cuando Google Translator falla
"""
import logging
from typing import Dict

class SimpleTranslator:
    """Traductor básico para demostración"""
    
    def __init__(self):
        # Diccionario básico de traducciones comunes
        self.translations = {
            "the": "el/la",
            "this": "este/esta",
            "video": "video",
            "purpose": "propósito",
            "introduce": "presentar",
            "course": "curso",
            "lab": "laboratorio",
            "environment": "entorno",
            "module": "módulo",
            "modules": "módulos",
            "hands-on": "práctico",
            "associated": "asociado",
            "watching": "viendo",
            "click": "hacer clic",
            "link": "enlace",
            "bring": "llevar",
            "service": "servicio",
            "free": "gratis",
            "use": "usar",
            "login": "iniciar sesión",
            "account": "cuenta",
            "accounts": "cuentas",
            "github": "github",
            "gitlab": "gitlab",
            "google": "google",
            "email": "correo electrónico",
            "platform": "plataforma",
            "loading": "cargando",
            "notice": "observar",
            "message": "mensaje",
            "configuring": "configurando",
            "important": "importante",
            "remember": "recordar",
            "complex": "complejo",
            "require": "requerir",
            "starting": "iniciando",
            "cluster": "clúster",
            "time": "tiempo",
            "load": "cargar",
            "wait": "esperar",
            "until": "hasta",
            "ready": "listo",
            "appears": "aparece",
            "skipped": "saltado",
            "ahead": "adelante",
            "command": "comando",
            "issued": "emitido",
            "user": "usuario",
            "shell": "shell",
            "logged": "conectado",
            "fully": "completamente",
            "capable": "capaz",
            "run": "ejecutar",
            "commands": "comandos",
            "experiment": "experimentar",
            "instructions": "instrucciones",
            "first": "primero",
            "instruction": "instrucción",
            "says": "dice",
            "start": "iniciar",
            "connecting": "conectando",
            "type": "escribir",
            "home": "inicio",
            "address": "dirección",
            "connected": "conectado",
            "tools": "herramientas",
            "available": "disponible",
            "look": "mirar",
            "top": "parte superior",
            "screen": "pantalla",
            "editor": "editor",
            "tab": "pestaña",
            "open": "abrir",
            "access": "acceso",
            "file": "archivo",
            "system": "sistema",
            "directory": "directorio",
            "underneath": "debajo",
            "three": "tres",
            "nodes": "nodos",
            "running": "ejecutándose",
            "during": "durante",
            "labs": "laboratorios",
            "directories": "directorios",
            "likewise": "igualmente",
            "said": "dijo",
            "one": "uno",
            "plus": "más",
            "sign": "signo",
            "new": "nuevo",
            "check": "verificar",
            "who": "quién",
            "root": "root",
            "useful": "útil",
            "well": "bien",
            "most": "mayoría",
            "here": "aquí",
            "execute": "ejecutar",
            "creating": "creando",
            "key": "clave",
            "space": "espacio",
            "over": "sobre",
            "real": "real",
            "enter": "ingresar",
            "follow": "seguir",
            "exactly": "exactamente",
            "shown": "mostrado",
            "left": "izquierda",
            "done": "terminado",
            "page": "página",
            "data": "datos",
            "next": "siguiente",
            "brings": "lleva",
            "end": "final",
            "multiple": "múltiples",
            "pages": "páginas",
            "instances": "instancias",
            "walk": "caminar",
            "either": "ya sea",
            "entering": "ingresando",
            "see": "ver",
            "them": "ellos",
            "interactive": "interactivo",
            "way": "manera",
            "want": "querer",
            "these": "estos",
            "note": "nota",
            "about": "acerca de",
            "only": "solo",
            "have": "tener",
            "calls": "llama",
            "scenario": "escenario",
            "hour": "hora",
            "same": "mismo",
            "happen": "pasar",
            "try": "intentar",
            "already": "ya",
            "make": "hacer",
            "sure": "seguro",
            "information": "información",
            "should": "debería",
            "enough": "suficiente",
            "get": "obtener",
            "up": "arriba",
            "our": "nuestro",
            "with": "con",
            "that": "que",
            "will": "será",
            "let": "permitir",
            "you": "tú/usted",
            "continue": "continuar",
            "thank": "gracias",
        }
    
    def translate_text(self, text: str) -> str:
        """
        Traduce texto básico palabra por palabra para demostración
        """
        if not text:
            return text
            
        words = text.lower().split()
        translated_words = []
        
        for word in words:
            # Limpiar puntuación
            clean_word = word.strip('.,!?;:"()[]{}')
            punctuation = word[len(clean_word):]
            
            # Buscar traducción
            if clean_word in self.translations:
                translated_word = self.translations[clean_word]
                translated_words.append(translated_word + punctuation)
            else:
                # Si no hay traducción, marcar como [EN:palabra]
                translated_words.append(f"[EN:{clean_word}]{punctuation}")
        
        return " ".join(translated_words)

def translate_to_spanish_demo(text: str) -> str:
    """
    Función de demostración para traducir a español
    """
    # Traducciones específicas para frases comunes del video
    translations = {
        "The purpose of this video is to introduce you to the lab environment for this course": 
        "El propósito de este video es presentarte el entorno de laboratorio para este curso",
        
        "Most of the modules in this course have a hands-on lab associated with them":
        "La mayoría de los módulos en este curso tienen un laboratorio práctico asociado",
        
        "After watching the module video, you can go down and click on the launch lab link":
        "Después de ver el video del módulo, puedes ir abajo y hacer clic en el enlace de lanzar laboratorio",
        
        "The lab environment for this class is a hosted service called Killecoda":
        "El entorno de laboratorio para esta clase es un servicio alojado llamado Killecoda",
        
        "You'll notice the message configuring the lab environment":
        "Notarás el mensaje configurando el entorno de laboratorio",
        
        "It's important to remember that some of the labs are complex":
        "Es importante recordar que algunos de los laboratorios son complejos",
        
        "So it does take some time for some of the labs to load up":
        "Así que toma algo de tiempo para que algunos laboratorios se carguen",
        
        "You're going to wait until the lab environment ready message appears":
        "Vas a esperar hasta que aparezca el mensaje de entorno de laboratorio listo",
        
        "This is fully capable bash shell and you can run commands":
        "Este es un shell bash completamente funcional y puedes ejecutar comandos",
        
        "The first instruction says start by connecting to the cluster":
        "La primera instrucción dice empezar conectándose al clúster",
        
        "Thank you": "Gracias"
    }
    
    # Buscar traducción exacta primero
    for english, spanish in translations.items():
        if english.lower() in text.lower():
            text = text.replace(english, spanish)
    
    # Si no hay traducción específica, usar traductor simple
    if text == text:  # No cambió mucho
        translator = SimpleTranslator()
        # Traducir solo las primeras palabras para demostrar
        words = text.split()
        if len(words) > 10:
            first_part = " ".join(words[:10])
            rest = " ".join(words[10:])
            translated_first = translator.translate_text(first_part)
            return f"{translated_first} [RESTO_EN_INGLÉS: {rest[:50]}...]"
        else:
            return translator.translate_text(text)
    
    return text