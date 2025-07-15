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

class AudioTranscriptionRequest(BaseModel):
    language: Optional[str] = Field(default="auto", description="Language code or 'auto' for detection")
    model_size: Optional[str] = Field(default="base", description="Whisper model size")
    translate_to: Optional[str] = Field(default=None, description="Target language for translation")
    quality_evaluation: Optional[bool] = Field(default=True, description="Enable quality evaluation")

class RealTimeTranscriptionRequest(BaseModel):
    language: Optional[str] = Field(default="auto", description="Language code or 'auto' for detection")
    model_size: Optional[str] = Field(default="tiny", description="Whisper model size - use 'tiny' for real-time")
    translate_to: Optional[str] = Field(default=None, description="Target language for real-time translation")
    chunk_duration: Optional[int] = Field(default=5, description="Audio chunk duration in seconds (1-10)")
    enable_vad: Optional[bool] = Field(default=True, description="Enable Voice Activity Detection")
    min_speech_duration: Optional[float] = Field(default=0.5, description="Minimum speech duration to process")
    silence_timeout: Optional[int] = Field(default=2, description="Silence timeout in seconds")

class RealTimeTranscriptionChunk(BaseModel):
    chunk_id: str
    session_id: str
    audio_data: str  # Base64 encoded audio
    timestamp: datetime
    is_final: bool = False

class RealTimeTranscriptionResponse(BaseModel):
    session_id: str
    chunk_id: str
    transcription: str
    translation: Optional[str] = None
    detected_language: Optional[str] = None
    confidence: Optional[float] = None
    is_final: bool
    processing_time: float
    timestamp: datetime

class RealTimeSession(BaseModel):
    session_id: str
    status: str  # "active", "paused", "completed", "error"
    language: str
    model_size: str
    translate_to: Optional[str] = None
    total_chunks: int = 0
    total_duration: float = 0.0
    full_transcription: str = ""
    full_translation: Optional[str] = None
    created_at: datetime
    last_activity: datetime
    error: Optional[str] = None

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

class AudioInfo(BaseModel):
    duration: float
    size: int
    audio_codec: Optional[str]
    sample_rate: Optional[int]
    channels: Optional[int]
    bitrate: Optional[int]

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
    source_type: Optional[str] = None  # "file", "url", or "audio"
    source_filename: Optional[str] = None  # For file uploads
    source_url: Optional[str] = None  # For URL downloads
    video_info: Optional[VideoInfo] = None
    audio_info: Optional[AudioInfo] = None  # For audio-only files
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
    duration: Optional[float] = None  # Audio/Video duration
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
