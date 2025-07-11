import re
import logging
from typing import Dict, List
from app.models.schemas import QualityReport, QualityMetrics

class QualityEvaluator:
    def __init__(self):
        self.quality_thresholds = {
            'excellent': 0.95,
            'good': 0.85,
            'acceptable': 0.70,
            'poor': 0.50
        }
    
    def evaluate_transcription(self, 
                             transcription_result: Dict,
                             audio_duration: float) -> QualityReport:
        """Evalúa la calidad de la transcripción"""
        try:
            logging.info("Evaluating transcription quality")
            
            # Validación inicial mejorada
            text = transcription_result.get('text', '')
            segments = transcription_result.get('segments', [])
            
            # Manejo de casos edge
            if not segments:
                return self._create_poor_quality_report("No transcription segments found")
            
            if not text or len(text.strip()) == 0:
                return self._create_poor_quality_report("Empty transcription text")
            
            # Calcular métricas individuales mejoradas
            confidence_score = self._calculate_improved_confidence(segments)
            word_count = len(text.split()) if text else 0
            speech_rate = self._calculate_speech_rate(text, audio_duration)
            silence_ratio = self._calculate_silence_ratio(segments, audio_duration)
            repetition_score = self._calculate_repetition_score(text)
            language_consistency = self._check_language_consistency(segments)
            
            metrics = QualityMetrics(
                confidence_score=confidence_score,
                word_count=word_count,
                speech_rate=speech_rate,
                silence_ratio=silence_ratio,
                repetition_score=repetition_score,
                language_consistency=language_consistency
            )
            
            # Calcular puntuación general mejorada
            overall_score = self._calculate_overall_score_improved(metrics, audio_duration, len(segments))
            quality_level = self._get_quality_level_improved(overall_score, metrics)
            recommendations = self._generate_recommendations_improved(metrics, overall_score, audio_duration)
            
            report = QualityReport(
                overall_score=overall_score,
                quality_level=quality_level,
                metrics=metrics,
                recommendations=recommendations
            )
            
            logging.info(f"Quality evaluation completed. Score: {overall_score:.2f}, Level: {quality_level}")
            return report
            
        except Exception as e:
            logging.error(f"Quality evaluation error: {e}")
            # Retornar reporte de error
            return QualityReport(
                overall_score=0.0,
                quality_level='error',
                metrics=QualityMetrics(
                    confidence_score=0.0,
                    word_count=0,
                    speech_rate=0.0,
                    silence_ratio=0.0,
                    repetition_score=0.0,
                    language_consistency=0.0
                ),
                recommendations=['Error evaluating quality']
            )
    
    def _calculate_speech_rate(self, text: str, duration: float) -> float:
        """Calcula palabras por minuto"""
        if duration <= 0 or not text:
            return 0.0
        
        word_count = len(text.split())
        return (word_count / duration) * 60
    
    def _calculate_silence_ratio(self, segments: List[Dict], total_duration: float) -> float:
        """Calcula ratio de silencio"""
        if not segments or total_duration <= 0:
            return 0.0
        
        speech_duration = sum(
            segment.get('end', 0) - segment.get('start', 0) 
            for segment in segments
        )
        
        return max(0.0, 1.0 - (speech_duration / total_duration))
    
    def _calculate_repetition_score(self, text: str) -> float:
        """Detecta repeticiones excesivas"""
        if not text or len(text.split()) < 10:
            return 1.0
        
        words = text.lower().split()
        
        # Contar repeticiones de frases de 3 palabras
        if len(words) >= 3:
            trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
            unique_trigrams = set(trigrams)
            repetition_ratio = len(unique_trigrams) / len(trigrams) if trigrams else 1.0
        else:
            # Para textos muy cortos, usar palabras únicas
            unique_words = set(words)
            repetition_ratio = len(unique_words) / len(words) if words else 1.0
        
        return repetition_ratio
    
    def _check_language_consistency(self, segments: List[Dict]) -> float:
        """Verifica consistencia del idioma detectado"""
        if not segments:
            return 1.0
        
        # Implementación simplificada
        # En una versión más avanzada, se podría usar detección de idioma por segmento
        return 0.9
    
    def _calculate_overall_score(self, metrics: QualityMetrics) -> float:
        """Calcula puntuación general ponderada"""
        weights = {
            'confidence_score': 0.4,
            'repetition_score': 0.2,
            'language_consistency': 0.2,
            'speech_rate_score': 0.2
        }
        
        # Normalizar speech rate (120-180 wpm es óptimo)
        speech_rate = metrics.speech_rate
        if 120 <= speech_rate <= 180:
            speech_rate_score = 1.0
        elif speech_rate == 0:
            speech_rate_score = 0.0
        else:
            # Penalizar desviaciones del rango óptimo
            optimal_center = 150
            deviation = abs(speech_rate - optimal_center)
            speech_rate_score = max(0.0, 1.0 - (deviation / optimal_center))
        
        # Calcular puntuación ponderada
        score = (
            metrics.confidence_score * weights['confidence_score'] +
            metrics.repetition_score * weights['repetition_score'] +
            metrics.language_consistency * weights['language_consistency'] +
            speech_rate_score * weights['speech_rate_score']
        )
        
        return min(1.0, max(0.0, score))
    
    def _get_quality_level(self, score: float) -> str:
        """Determina nivel de calidad"""
        for level, threshold in self.quality_thresholds.items():
            if score >= threshold:
                return level
        return 'poor'
    
    def _create_poor_quality_report(self, error_message: str) -> QualityReport:
        """Crea un reporte de calidad para casos de error o calidad muy baja."""
        return QualityReport(
            overall_score=0.0,
            quality_level='poor',
            metrics=QualityMetrics(
                confidence_score=0.0,
                word_count=0,
                speech_rate=0.0,
                silence_ratio=1.0,
                repetition_score=0.0,
                language_consistency=0.0
            ),
            recommendations=[error_message, "Please check your audio quality and try again"]
        )
    
    def _calculate_improved_confidence(self, segments: List[Dict]) -> float:
        """Calcula confianza promedio mejorada con manejo de valores faltantes."""
        if not segments:
            return 0.0
        
        valid_confidences = []
        for segment in segments:
            confidence = segment.get('avg_logprob') or segment.get('confidence')
            if confidence is not None:
                # Convertir avg_logprob (negativo) a confianza (0-1)
                if confidence <= 0:
                    confidence = min(1.0, max(0.0, 1.0 + confidence))
                valid_confidences.append(confidence)
        
        if not valid_confidences:
            # Si no hay valores de confianza, usar estimación basada en longitud de texto
            avg_text_length = sum(len(seg.get('text', '')) for seg in segments) / len(segments)
            return min(0.8, avg_text_length / 50.0)  # Heurística simple
        
        return sum(valid_confidences) / len(valid_confidences)
    
    def _calculate_overall_score_improved(self, metrics: QualityMetrics, audio_duration: float, segment_count: int) -> float:
        """Calcula puntuación general mejorada con factores adicionales."""
        weights = {
            'confidence_score': 0.35,
            'repetition_score': 0.20,
            'language_consistency': 0.15,
            'speech_rate_score': 0.15,
            'density_score': 0.10,
            'segment_quality': 0.05
        }
        
        # Normalizar speech rate (120-180 wpm es óptimo)
        speech_rate = metrics.speech_rate
        if 120 <= speech_rate <= 180:
            speech_rate_score = 1.0
        elif speech_rate == 0:
            speech_rate_score = 0.0
        else:
            optimal_center = 150
            deviation = abs(speech_rate - optimal_center)
            speech_rate_score = max(0.0, 1.0 - (deviation / optimal_center))
        
        # Score de densidad de contenido
        if audio_duration > 0:
            words_per_second = metrics.word_count / audio_duration
            density_score = min(1.0, words_per_second / 3.0)  # 3 palabras/segundo es bueno
        else:
            density_score = 0.0
        
        # Score de calidad de segmentos
        if segment_count > 0:
            avg_words_per_segment = metrics.word_count / segment_count
            segment_quality = min(1.0, avg_words_per_segment / 10.0)  # 10 palabras/segmento ideal
        else:
            segment_quality = 0.0
        
        # Calcular puntuación ponderada
        score = (
            metrics.confidence_score * weights['confidence_score'] +
            metrics.repetition_score * weights['repetition_score'] +
            metrics.language_consistency * weights['language_consistency'] +
            speech_rate_score * weights['speech_rate_score'] +
            density_score * weights['density_score'] +
            segment_quality * weights['segment_quality']
        )
        
        return min(1.0, max(0.0, score))
    
    def _get_quality_level_improved(self, score: float, metrics: QualityMetrics) -> str:
        """Determina nivel de calidad con consideraciones adicionales."""
        # Penalizaciones por condiciones específicas
        penalties = 0
        
        if metrics.confidence_score < 0.5:
            penalties += 0.1
        
        if metrics.word_count < 5:
            penalties += 0.15
        
        if metrics.silence_ratio > 0.8:
            penalties += 0.1
        
        adjusted_score = max(0.0, score - penalties)
        
        # Determinar nivel basado en score ajustado
        if adjusted_score >= 0.90:
            return 'excellent'
        elif adjusted_score >= 0.75:
            return 'good'
        elif adjusted_score >= 0.60:
            return 'acceptable'
        elif adjusted_score >= 0.40:
            return 'poor'
        else:
            return 'very_poor'
    
    def _generate_recommendations_improved(self, metrics: QualityMetrics, overall_score: float, audio_duration: float) -> List[str]:
        """Genera recomendaciones mejoradas y específicas."""
        recommendations = []
        
        # Recomendaciones por confianza
        if metrics.confidence_score < 0.6:
            recommendations.append("Use a larger Whisper model (medium/large) for better accuracy")
            recommendations.append("Ensure audio quality is high (clear speech, minimal background noise)")
        elif metrics.confidence_score < 0.8:
            recommendations.append("Consider using a medium Whisper model for improved accuracy")
        
        # Recomendaciones por duración y contenido
        if audio_duration > 0:
            words_per_minute = (metrics.word_count / audio_duration) * 60
            if words_per_minute < 80:
                recommendations.append("Speech rate is very slow - verify audio contains clear speech")
            elif words_per_minute > 200:
                recommendations.append("Speech rate is very fast - consider using a larger model for better accuracy")
        
        # Recomendaciones por silencio
        if metrics.silence_ratio > 0.6:
            recommendations.append("High silence ratio - consider trimming silent portions or check audio extraction")
        
        # Recomendaciones por repetición
        if metrics.repetition_score < 0.6:
            recommendations.append("High repetition detected - review transcription for accuracy")
            recommendations.append("Consider using post-processing to remove duplicate content")
        
        # Recomendaciones por longitud
        if metrics.word_count < 10:
            recommendations.append("Very short transcription - verify video contains speech content")
        elif metrics.word_count > 5000:
            recommendations.append("Long transcription - consider processing in segments for better accuracy")
        
        # Recomendaciones por score general
        if overall_score < 0.5:
            recommendations.append("Overall quality is low - review audio source and processing parameters")
        
        return recommendations if recommendations else ["Transcription quality meets acceptable standards"]
    
    def _generate_recommendations(self, metrics: QualityMetrics) -> List[str]:
        """Genera recomendaciones para mejorar calidad"""
        recommendations = []
        
        if metrics.confidence_score < 0.8:
            recommendations.append(
                "Consider using a larger Whisper model (small, medium, or large) for better accuracy"
            )
        
        if metrics.silence_ratio > 0.5:
            recommendations.append(
                "High silence ratio detected - consider audio preprocessing or check for audio issues"
            )
        
        speech_rate = metrics.speech_rate
        if speech_rate < 100 and speech_rate > 0:
            recommendations.append(
                "Speech rate is very slow - check for audio quality issues"
            )
        elif speech_rate > 250:
            recommendations.append(
                "Speech rate is very fast - transcription might be less accurate"
            )
        elif speech_rate == 0:
            recommendations.append(
                "No speech detected - verify audio extraction was successful"
            )
        
        if metrics.repetition_score < 0.7:
            recommendations.append(
                "High repetition detected - review transcription carefully for accuracy"
            )
        
        if metrics.word_count < 10:
            recommendations.append(
                "Very short transcription - verify video contains speech"
            )
        
        return recommendations if recommendations else ["Transcription quality is good"]
