from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

class TranscriptionRequest(BaseModel):
    language: Optional[str] = Field(default="auto", description="Language code or 'auto' for detection")
    model_size: Optional[str] = Field(default="base", description="Whisper model size")
    translate_to: Optional[str] = Field(default=None, description="Target language for translation")
    quality_evaluation: Optional[bool] = Field(default=True, description="Enable quality evaluation")

class TranscriptionSegment(BaseModel):
    id: int
    start: float
    end: float
    text: str
    confidence: Optional[float] = None

class QualityMetrics(BaseModel):
    confidence_score: float
    word_count: int
    speech_rate: float
    silence_ratio: float
    repetition_score: float
    language_consistency: float

class QualityReport(BaseModel):
    overall_score: float
    quality_level: str
    metrics: QualityMetrics
    recommendations: List[str]

class VideoInfo(BaseModel):
    duration: float
    size: int
    video_codec: Optional[str]
    audio_codec: Optional[str]
    fps: Optional[float]

class TranscriptionResult(BaseModel):
    job_id: str
    status: str
    video_info: Optional[VideoInfo] = None
    transcription_text: Optional[str] = None
    translation_text: Optional[str] = None
    detected_language: Optional[str] = None
    segments: Optional[List[TranscriptionSegment]] = None
    translation_segments: Optional[List[TranscriptionSegment]] = None
    quality_report: Optional[QualityReport] = None
    srt_file_path: Optional[str] = None
    processing_time: Optional[float] = None
    created_at: datetime
    completed_at: Optional[datetime] = None

class ErrorResponse(BaseModel):
    error: str
    detail: str
    job_id: Optional[str] = None
    error_code: Optional[str] = None
    timestamp: Optional[str] = None
    suggestions: Optional[List[str]] = None

class ValidationErrorResponse(BaseModel):
    error: str = "Validation Error"
    detail: str
    validation_errors: List[dict]
    timestamp: str

class ProcessingStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: Optional[float] = None  # 0.0 to 1.0
    current_step: Optional[str] = None
    estimated_completion: Optional[str] = None
    error: Optional[str] = None
