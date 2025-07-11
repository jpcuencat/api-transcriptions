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
    
    def generate_srt(self, segments: List[Dict], output_path: str) -> str:
        """Genera archivo SRT desde segmentos de Whisper"""
        try:
            logging.info(f"Generating SRT file: {output_path}")
            logging.info(f"Processing {len(segments)} segments")
            
            if not segments:
                raise Exception("No segments provided for SRT generation")
            
            # Optimizar segmentos para mejor legibilidad
            optimized_segments = self._optimize_segments(segments)
            
            # Formatear como SRT
            srt_content = self._format_srt(optimized_segments)
            
            # Guardar archivo
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(srt_content)
            
            logging.info(f"SRT file generated successfully: {output_path}")
            return output_path
            
        except Exception as e:
            logging.error(f"SRT generation error: {e}")
            raise Exception(f"SRT generation failed: {e}")
    
    def _optimize_segments(self, segments: List[Dict]) -> List[Dict]:
        """Optimiza segmentos para mejor legibilidad"""
        optimized = []
        
        for i, segment in enumerate(segments):
            text = segment.get('text', '').strip()
            if not text:
                continue
                
            start = segment.get('start', 0)
            end = segment.get('end', start + 2)
            duration = end - start
            
            # Ajustar duración mínima/máxima
            if duration < self.min_duration:
                end = start + self.min_duration
            elif duration > self.max_duration:
                end = start + self.max_duration
            
            # Asegurar gap mínimo con siguiente segmento
            if i < len(segments) - 1:
                next_start = segments[i + 1].get('start', end + 1)
                if end + self.min_gap > next_start:
                    end = max(start + self.min_duration, next_start - self.min_gap)
            
            # Dividir texto largo en múltiples líneas
            lines = self._split_text_into_lines(text)
            
            optimized.append({
                'text': '\n'.join(lines),
                'start': start,
                'end': end
            })
        
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
