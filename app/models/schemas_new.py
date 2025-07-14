from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

class TranscriptionRequest(BaseModel):
    language: Optional[str] = Field(default="auto", description="Language code or 'auto' for detection")
    model_size: Optional[str] = Field(default="base", description="Whisper model size")
    translate_to: Optional[str] = Field(default=None, description="Target language for translation")
    quality_evaluation: Optional[bool] = Field(default=True, description="Enable quality evaluation")

class URLTranscriptionRequest(BaseModel):
    url: str = Field(..., description="URL of the video to transcribe")
    language: Optional[str] = Field(default="auto", description="Language code or 'auto' for detection")
    model_size: Optional[str] = Field(default="base", description="Whisper model size")
    translate_to: Optional[str] = Field(default=None, description="Target language for translation")
    quality_evaluation: Optional[bool] = Field(default=True, description="Enable quality evaluation")
    video_quality: Optional[str] = Field(default="medium", description="Video download quality: low, medium, best")

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

class VideoUrlInfo(BaseModel):
    title: str
    duration: int
    uploader: str
    view_count: Optional[int] = None
    upload_date: str
    description: str
    thumbnail: Optional[str] = None
    webpage_url: str
    extractor: str
    has_audio: bool
    has_video: bool

class URLValidationResult(BaseModel):
    valid: bool
    accessible: bool
    video_info: Optional[VideoUrlInfo] = None
    warnings: List[str] = []
    errors: List[str] = []

class TranscriptionResult(BaseModel):
    job_id: str
    status: str
    source_type: Optional[str] = None  # "file" or "url"
    source_filename: Optional[str] = None  # For file uploads
    source_url: Optional[str] = None  # For URL downloads
    video_info: Optional[VideoInfo] = None
    transcription_text: Optional[str] = None
    translated_text: Optional[str] = None  # Field for translation text
    detected_language: Optional[str] = None
    language: Optional[str] = None  # Detected language field
    target_language: Optional[str] = None  # Translation target language
    model_used: Optional[str] = None  # Model used for transcription
    segments: Optional[List[TranscriptionSegment]] = None
    translation_segments: Optional[List[TranscriptionSegment]] = None
    quality_report: Optional[QualityReport] = None
    srt_file_path: Optional[str] = None
    processing_time: Optional[float] = None
    duration: Optional[float] = None  # Video duration
    error: Optional[str] = None  # Error message if failed
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
