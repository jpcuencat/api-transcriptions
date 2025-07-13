import re
import os
import logging
from typing import List, Dict
from datetime import timedelta

class SubtitleGenerator:
    def __init__(self):
        self.max_chars_per_line = 42
        self.max_lines_per_subtitle = 2
        self.min_duration = 1.0  # segundos
        self.max_duration = 7.0  # segundos
        self.min_gap = 0.5       # gap mínimo entre subtítulos
    
    def generate_srt(self, segments: List[Dict], output_path: str, use_translated: bool = False) -> str:
        """Genera archivo SRT desde segmentos de Whisper"""
        try:
            logging.info(f"Generating SRT file: {output_path}")
            logging.info(f"Processing {len(segments)} segments (use_translated: {use_translated})")

            if not output_path:
                raise ValueError("Output path cannot be empty")

            if not segments:
                raise Exception("No segments provided for SRT generation")

            # Preparar segmentos con el texto correcto
            processed_segments = []
            for i, segment in enumerate(segments):
                if use_translated and 'translation' in segment:
                    # Usar el texto traducido
                    text = segment['translation']
                    logging.debug(f"Using translated text for segment {i}: '{text[:50]}...'")
                else:
                    # Usar el texto original
                    text = segment.get('text', '')
                    logging.debug(f"Using original text for segment {i}: '{text[:50]}...'")
                
                if text.strip():  # Solo procesar segmentos con texto
                    processed_segment = {
                        'text': text.strip(),
                        'start': segment.get('start', 0),
                        'end': segment.get('end', segment.get('start', 0) + 2)
                    }
                    processed_segments.append(processed_segment)

            logging.info(f"Processed {len(processed_segments)} valid segments")

            # Optimizar segmentos para mejor legibilidad
            optimized_segments = self._optimize_segments(processed_segments)

            # Formatear como SRT
            srt_content = self._format_srt(optimized_segments)

            # Guardar archivo
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(srt_content)

            logging.info(f"SRT file generated successfully: {output_path} with {len(optimized_segments)} subtitles")
            return output_path

        except ValueError as ve:
            logging.error(f"Value error: {ve}")
            raise
        except Exception as e:
            logging.error(f"SRT generation error: {e}")
            raise Exception(f"SRT generation failed: {e}")
    
    def generate_srt_from_full_text(self, full_text: str, total_duration: float, output_path: str) -> str:
        """Genera archivo SRT a partir del texto completo dividido en segmentos temporales apropiados"""
        try:
            logging.info(f"Generating SRT from full text: {len(full_text)} characters, duration: {total_duration}s")
            logging.info(f"Full text preview: '{full_text[:100]}...'")

            if not output_path:
                raise ValueError("Output path cannot be empty")

            if not full_text or not full_text.strip():
                raise Exception("No text provided for SRT generation")

            if total_duration <= 0:
                total_duration = 60.0  # Duración por defecto de 1 minuto

            # Dividir el texto completo en oraciones y crear segmentos temporales
            segments = self._create_segments_from_text(full_text, total_duration)
            
            logging.info(f"Created {len(segments)} segments from full text")

            # Formatear como SRT
            srt_content = self._format_srt(segments)

            # Guardar archivo
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(srt_content)

            logging.info(f"SRT file generated successfully from full text: {output_path} with {len(segments)} subtitles")
            return output_path

        except ValueError as ve:
            logging.error(f"Value error: {ve}")
            raise
        except Exception as e:
            logging.error(f"SRT generation from full text error: {e}")
            raise Exception(f"SRT generation from full text failed: {e}")
    
    def _create_segments_from_text(self, text: str, total_duration: float) -> List[Dict]:
        """Crea segmentos temporales a partir del texto completo"""
        
        # Dividir el texto en oraciones
        sentences = self._split_into_sentences(text)
        
        if not sentences:
            return []
        
        # Calcular duración promedio por carácter
        total_chars = len(text)
        chars_per_second = total_chars / total_duration if total_duration > 0 else 10
        
        logging.info(f"Text analysis: {total_chars} chars, {total_duration}s, {chars_per_second:.2f} chars/sec")
        
        segments = []
        current_time = 0.0
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Calcular duración basada en longitud del texto
            sentence_chars = len(sentence)
            estimated_duration = max(self.min_duration, sentence_chars / chars_per_second)
            estimated_duration = min(self.max_duration, estimated_duration)
            
            # Ajustar si es la última oración para que termine con la duración total
            if i == len(sentences) - 1:
                end_time = total_duration
            else:
                end_time = current_time + estimated_duration
            
            # Dividir oraciones largas en múltiples líneas para subtítulos
            subtitle_lines = self._split_text_into_lines(sentence)
            
            segment = {
                'text': '\n'.join(subtitle_lines),
                'start': current_time,
                'end': end_time
            }
            
            segments.append(segment)
            logging.debug(f"Segment {i}: {current_time:.2f}-{end_time:.2f}s, '{sentence[:30]}...'")
            
            # Actualizar tiempo para el siguiente segmento
            current_time = end_time + self.min_gap
            
            # Asegurar que no excedamos la duración total
            if current_time >= total_duration:
                break
        
        logging.info(f"Generated {len(segments)} segments covering {current_time:.2f}s of {total_duration}s")
        return segments
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Divide el texto en oraciones"""
        # Usar regex para dividir en oraciones respetando puntos, exclamaciones y preguntas
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Filtrar oraciones vacías y muy cortas
        valid_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 3:  # Mínimo 3 caracteres
                valid_sentences.append(sentence)
        
        # Si no hay oraciones válidas, dividir por comas o puntos y comas
        if not valid_sentences:
            sentences = re.split(r'[,;]\s+', text)
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 10:  # Mínimo 10 caracteres para fragmentos por coma
                    valid_sentences.append(sentence)
        
        # Si aún no hay oraciones válidas, usar el texto completo
        if not valid_sentences:
            valid_sentences = [text.strip()]
        
        return valid_sentences
    
    def _optimize_segments(self, segments: List[Dict]) -> List[Dict]:
        """Optimiza segmentos manteniendo TODOS los segmentos originales de Whisper"""
        logging.info(f"Optimizing {len(segments)} segments - PRESERVING ALL SEGMENTS")
        optimized = []
        
        for i, segment in enumerate(segments):
            text = segment.get('text', '').strip()
            start = segment.get('start', 0)
            end = segment.get('end', start + 1.0)
            
            # CRÍTICO: Conservar TODOS los segmentos, incluso los vacíos
            if not text:
                logging.debug(f"Preserving empty segment {i}: start={start}, end={end}")
                # Crear un segmento vacío pero válido para mantener la estructura temporal
                optimized.append({
                    'text': "",  # Segmento vacío
                    'start': start,
                    'end': end
                })
                continue
                
            # Para segmentos con texto, mantener timing original de Whisper
            if end <= start:
                end = start + 1.0  # Asegurar duración mínima
            
            logging.debug(f"Processing segment {i}: start={start}, end={end}, text='{text[:50]}...'")
            
            # Dividir texto largo pero mantener como un solo subtítulo para conservar timing
            lines = self._split_text_into_lines(text)
            
            optimized.append({
                'text': '\n'.join(lines),
                'start': start,
                'end': end
            })
        
        logging.info(f"Segment optimization complete: {len(segments)} original -> {len(optimized)} optimized (PRESERVED ALL)")
        
        # Verificación final: asegurar que tenemos el mismo número de segmentos
        if len(optimized) != len(segments):
            logging.error(f"SEGMENT LOSS DETECTED! Original: {len(segments)}, Optimized: {len(optimized)}")
            # En caso de discrepancia, usar los segmentos originales
            return segments
        
        return optimized
    
    def _split_text_into_lines(self, text: str) -> List[str]:
        """Divide texto en líneas apropiadas para subtítulos"""
        if len(text) <= self.max_chars_per_line:
            return [text]
        
        # Dividir por palabras
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            test_line = current_line + " " + word if current_line else word
            
            if len(test_line) <= self.max_chars_per_line:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                    current_line = word
                else:
                    # Palabra muy larga, dividir por fuerza
                    lines.append(word[:self.max_chars_per_line])
                    current_line = word[self.max_chars_per_line:]
        
        if current_line:
            lines.append(current_line)
        
        # Limitar a máximo de líneas por subtítulo
        if len(lines) > self.max_lines_per_subtitle:
            # Combinar líneas excedentes
            combined_lines = lines[:self.max_lines_per_subtitle-1]
            combined_lines.append(' '.join(lines[self.max_lines_per_subtitle-1:]))
            lines = combined_lines
        
        return lines
    
    def _format_srt(self, segments: List[Dict]) -> str:
        """Formatea segmentos como SRT"""
        srt_content = ""
        
        for i, segment in enumerate(segments, 1):
            start_time = self._seconds_to_srt_time(segment['start'])
            end_time = self._seconds_to_srt_time(segment['end'])
            text = segment['text']
            
            srt_content += f"{i}\n"
            srt_content += f"{start_time} --> {end_time}\n"
            srt_content += f"{text}\n\n"
        
        return srt_content.strip()
    
    def _seconds_to_srt_time(self, seconds: float) -> str:
        """Convierte segundos a formato SRT (HH:MM:SS,mmm)"""
        td = timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        milliseconds = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"
