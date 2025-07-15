import os
import json
import uuid
import time
import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Depends, Request
from app.models.schemas import (
    RealTimeTranscriptionRequest, 
    RealTimeTranscriptionResponse, 
    RealTimeSession,
    RealTimeTranscriptionChunk
)
from app.services.realtime_transcription_service_v3 import RealTimeTranscriptionServiceV3
from app.core.security import security_manager

router = APIRouter()

# Servicio de transcripción en tiempo real
realtime_service = RealTimeTranscriptionServiceV3()

@router.post("/create-session", response_model=RealTimeSession)
async def create_realtime_session(
    request: RealTimeTranscriptionRequest,
    req: Request
):
    """
    Crea una nueva sesión de transcripción en tiempo real para micrófono
    """
    try:
        # Verificar límites de usuario
        await security_manager.check_user_limits(req.client.host, "realtime_session")
        
        # Generar ID único para la sesión
        session_id = str(uuid.uuid4())
        
        # Crear sesión
        session = await realtime_service.create_session(
            session_id=session_id,
            language=request.language,
            model_size=request.model_size,
            translate_to=request.translate_to
        )
        
        logging.info(f"Created real-time session {session_id} for {req.client.host}")
        return session
        
    except Exception as e:
        logging.error(f"Error creating real-time session: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating session: {str(e)}")

@router.post("/process-chunk", response_model=Optional[RealTimeTranscriptionResponse])
async def process_realtime_chunk(
    chunk: RealTimeTranscriptionChunk,
    req: Request
):
    """
    Procesa un chunk de audio capturado del micrófono en tiempo real
    """
    try:
        # Verificar que la sesión existe
        session = await realtime_service.get_session(chunk.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Procesar chunk de audio del micrófono
        result = await realtime_service.process_audio_chunk(
            session_id=chunk.session_id,
            chunk_id=chunk.chunk_id,
            audio_base64=chunk.audio_data,
            enable_vad=True,
            min_speech_duration=0.5
        )
        
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.error(f"Error processing audio chunk: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@router.get("/session/{session_id}", response_model=RealTimeSession)
async def get_realtime_session(session_id: str):
    """
    Obtiene información de una sesión de tiempo real
    """
    session = await realtime_service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session

@router.post("/pause/{session_id}")
async def pause_realtime_session(session_id: str):
    """
    Pausa una sesión de transcripción en tiempo real
    """
    success = await realtime_service.pause_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "paused", "session_id": session_id}

@router.post("/resume/{session_id}")
async def resume_realtime_session(session_id: str):
    """
    Reanuda una sesión de transcripción en tiempo real
    """
    success = await realtime_service.resume_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found or not paused")
    return {"status": "resumed", "session_id": session_id}

@router.post("/close/{session_id}")
async def close_realtime_session(session_id: str):
    """
    Cierra una sesión de transcripción en tiempo real y devuelve transcripción completa
    """
    session = await realtime_service.close_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "status": "closed", 
        "session_id": session_id,
        "full_transcription": session.full_transcription,
        "full_translation": session.full_translation,
        "total_chunks": session.total_chunks,
        "total_duration": session.total_duration
    }

# ===== WEBSOCKET PARA TRANSCRIPCIÓN EN TIEMPO REAL =====

@router.websocket("/ws/{session_id}")
async def websocket_realtime_transcription(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint para transcripción en tiempo real de micrófono
    Permite streaming de audio desde el navegador al servidor
    """
    await websocket.accept()
    logging.info(f"WebSocket connection established for session {session_id}")
    
    try:
        # Verificar que la sesión existe
        session = await realtime_service.get_session(session_id)
        if not session:
            await websocket.send_text(json.dumps({
                "error": "Session not found",
                "session_id": session_id
            }))
            await websocket.close()
            return
        
        # Enviar confirmación de conexión
        await websocket.send_text(json.dumps({
            "type": "connection_established",
            "session_id": session_id,
            "status": session.status,
            "message": "Micrófono conectado. Comience a hablar..."
        }))
        
        while True:
            try:
                # Recibir mensaje del cliente (audio del micrófono)
                data = await websocket.receive_text()
                message = json.loads(data)
                
                message_type = message.get("type")
                
                if message_type == "audio_chunk":
                    # Procesar chunk de audio del micrófono
                    chunk_id = message.get("chunk_id", str(uuid.uuid4()))
                    audio_data = message.get("audio_data")
                    
                    if not audio_data:
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "message": "No se recibió audio del micrófono"
                        }))
                        continue
                    
                    # Procesar audio capturado con servicio simplificado
                    result = await realtime_service.process_audio_chunk(
                        session_id=session_id,
                        chunk_id=chunk_id,
                        audio_base64=audio_data
                    )
                    
                    if result:
                        # Enviar resultado de transcripción en tiempo real
                        await websocket.send_text(json.dumps({
                            "type": "transcription",  # Cambiado para coincidir con el frontend
                            "session_id": result.session_id,
                            "chunk_id": result.chunk_id,
                            "transcription": result.transcription,
                            "translation": result.translation,
                            "detected_language": result.detected_language,
                            "processing_time": result.processing_time,
                            "timestamp": result.timestamp.isoformat(),
                            "message": "Transcripción en tiempo real"
                        }))
                    else:
                        # No hay resultado (no se detectó voz o audio muy corto)
                        await websocket.send_text(json.dumps({
                            "type": "no_transcription",
                            "chunk_id": chunk_id,
                            "reason": "No se detectó voz o audio muy corto",
                            "message": "Esperando audio del micrófono..."
                        }))
                
                elif message_type == "get_session_info":
                    # Enviar información completa de la sesión
                    current_session = await realtime_service.get_session(session_id)
                    if current_session:
                        await websocket.send_text(json.dumps({
                            "type": "session_info",
                            "session": {
                                "session_id": current_session.session_id,
                                "status": current_session.status,
                                "language": current_session.language,
                                "translate_to": current_session.translate_to,
                                "total_chunks": current_session.total_chunks,
                                "total_duration": current_session.total_duration,
                                "full_transcription": current_session.full_transcription,
                                "full_translation": current_session.full_translation
                            }
                        }))
                
                elif message_type == "pause":
                    # Pausar captura de micrófono
                    success = await realtime_service.pause_session(session_id)
                    await websocket.send_text(json.dumps({
                        "type": "session_paused",
                        "success": success,
                        "message": "Captura de micrófono pausada"
                    }))
                
                elif message_type == "resume":
                    # Reanudar captura de micrófono
                    success = await realtime_service.resume_session(session_id)
                    await websocket.send_text(json.dumps({
                        "type": "session_resumed",
                        "success": success,
                        "message": "Captura de micrófono reanudada"
                    }))
                
                elif message_type == "close":
                    # Cerrar sesión y obtener transcripción final
                    final_session = await realtime_service.close_session(session_id)
                    if final_session:
                        await websocket.send_text(json.dumps({
                            "type": "session_closed",
                            "final_transcription": final_session.full_transcription,
                            "final_translation": final_session.full_translation,
                            "total_chunks": final_session.total_chunks,
                            "total_duration": final_session.total_duration,
                            "message": "Sesión cerrada. Transcripción completa lista."
                        }))
                    break
                
                else:
                    try:
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "message": f"Tipo de mensaje desconocido: {message_type}"
                        }))
                    except:
                        break
                    
            except json.JSONDecodeError:
                try:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": "Formato JSON inválido"
                    }))
                except:
                    break
            except Exception as e:
                logging.error(f"Error in WebSocket processing: {e}")
                try:
                    await websocket.send_text(json.dumps({
                        "type": "error", 
                        "message": f"Error procesando audio: {str(e)}"
                    }))
                except:
                    break
    
    except WebSocketDisconnect:
        logging.info(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        logging.error(f"WebSocket error for session {session_id}: {e}")
    finally:
        # Siempre cerrar sesión al finalizar
        try:
            await realtime_service.close_session(session_id)
        except Exception as e:
            logging.error(f"Error closing session {session_id}: {e}")
