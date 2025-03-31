"""
Voice Agent - Real-time voice conversation interface using OpenAI APIs.

This module provides a complete voice-based conversation system that:
1. Captures audio from the microphone
2. Detects when the user starts and stops speaking
3. Transcribes speech using OpenAI's Whisper model
4. Generates responses using OpenAI's GPT model
5. Converts responses to speech using OpenAI's TTS
6. Handles interruptions and conversation flow

The system is designed to be responsive and natural, with
features like silence detection, speech interruption, and
graceful termination.
"""

import os
import sys
import asyncio
import tempfile
import audioop
import wave
import pyaudio
import time
import threading
import queue
import signal
import io
from typing import List, Optional, Callable, Any, Dict, Union, Tuple, Set
from dotenv import load_dotenv
from openai import AsyncOpenAI

# Local imports
try:
    import voice_agent_utils
    has_utils = True
except ImportError:
    has_utils = False

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Load environment variables
load_dotenv()

# Constants
class AudioConfig:
    """Audio recording and processing configuration"""
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 24000
    SILENCE_THRESHOLD = 1000  # Threshold for speech detection
    SILENCE_DURATION = 0.8  # Seconds of silence to end speaking
    CONVERSATION_TIMEOUT = 8.0  # Seconds of silence to end conversation
    INTERRUPTION_THRESHOLD = 1200  # Threshold for detecting interruptions

class TerminationConfig:
    """Configuration for conversation termination"""
    KEYWORDS = [
        "stop", "exit", "quit", "bye", "end", "terminate", "goodbye", 
        "stop now", "please stop", "that's all", "thank you", "thanks",
        "done", "no more", "enough", "shut down", "close", "all done"
    ]
    SIMPLE_WORDS = ["stop", "quit", "exit", "bye", "end"]
    
    @classmethod
    def check_for_termination(cls, text: str) -> bool:
        """Check if text contains termination intent
        
        Args:
            text: The text to check for termination keywords
            
        Returns:
            True if termination intent is detected, False otherwise
        """
        if not text:
            return False
        
        text = text.lower()
        print(f"Checking termination for: '{text}'")
        
        # Check for termination keywords
        for keyword in cls.KEYWORDS:
            if keyword in text:
                print(f"Termination match found: '{keyword}'")
                return True
        
        # Additionally check for any occurrence of simple stop words
        for word in cls.SIMPLE_WORDS:
            if f" {word}" in f" {text} ":  # Add spaces to ensure whole word matching
                print(f"Simple stop word found: '{word}'")
                return True
        
        return False

class AIConfig:
    """AI model configuration"""
    STT_MODEL = "whisper-1"
    LLM_MODEL = "gpt-4o"
    TTS_MODEL = "tts-1"
    TTS_VOICE = "nova"
    TEMPERATURE = 0.7
    SYSTEM_PROMPT = (
        "You are a helpful voice assistant. Keep your responses concise and "
        "natural for speech. Use informal language as if you're having a conversation. "
        "Always respond in English, regardless of the language used to speak to you."
    )

# Global flag for immediate termination
FORCE_STOP = False

# -----------------------------------------------------------------------------
# Voice Activity Detection
# -----------------------------------------------------------------------------

class VADAnalyzer:
    """Voice Activity Detection to determine when someone is speaking"""
    
    def __init__(self, threshold: int = AudioConfig.SILENCE_THRESHOLD, 
                 silence_duration: float = AudioConfig.SILENCE_DURATION):
        """Initialize the VAD analyzer
        
        Args:
            threshold: RMS threshold for speech detection
            silence_duration: How long silence must persist to end speech detection
        """
        self.threshold = threshold
        self.silence_duration = silence_duration
        self.speaking = False
        self.silence_start = None
        self.frames: List[bytes] = []
        
    def is_speech(self, audio_chunk: bytes) -> bool:
        """Determine if audio chunk contains speech
        
        Args:
            audio_chunk: Raw audio data to analyze
            
        Returns:
            True if speech is detected or we're in a speaking state,
            False if silence is detected for longer than silence_duration
        """
        rms = audioop.rms(audio_chunk, 2)
        
        # If volume above threshold, it's speech
        is_speech = rms > self.threshold
        
        current_time = time.time()
        
        if is_speech:
            self.speaking = True
            self.silence_start = None
            self.frames.append(audio_chunk)
            return True
        
        # If we were speaking but now detect silence
        if self.speaking:
            if self.silence_start is None:
                self.silence_start = current_time
                self.frames.append(audio_chunk)
                return True
                
            # If silence has lasted long enough, speaking has ended
            if current_time - self.silence_start > self.silence_duration:
                self.speaking = False
                return False
                
            # Still counting silence but not long enough to end
            self.frames.append(audio_chunk)
            return True
            
        # Not speaking and still silent
        return False
        
    def reset(self) -> List[bytes]:
        """Reset VAD state and get the collected frames
        
        Returns:
            The collected audio frames
        """
        frames = self.frames
        self.frames = []
        self.speaking = False
        self.silence_start = None
        return frames

# -----------------------------------------------------------------------------
# Audio Input Service
# -----------------------------------------------------------------------------

class AudioInputService:
    """Handle microphone input with voice activity detection"""
    
    def __init__(self):
        """Initialize the audio input service"""
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.vad = VADAnalyzer()
        self.running = False
        self.callback = None
        self.frames_queue = queue.Queue()
        self.is_ai_speaking = False
        self.is_generating_speech = False
        self.last_activity_time = time.time()
        self.audio_thread = None
        
    async def start(self, callback: Callable):
        """Start listening for audio
        
        Args:
            callback: Function to call when speech is detected
        """
        self.callback = callback
        self.running = True
        self.last_activity_time = time.time()
        
        # Open audio stream
        self.stream = self.audio.open(
            format=AudioConfig.FORMAT,
            channels=AudioConfig.CHANNELS,
            rate=AudioConfig.RATE,
            input=True,
            frames_per_buffer=AudioConfig.CHUNK
        )
        
        print("Listening for speech...")
        
        # Start a thread for audio processing
        self.audio_thread = threading.Thread(target=self._process_audio)
        self.audio_thread.daemon = True
        self.audio_thread.start()
        
        # Start a task to check for processed frames
        asyncio.create_task(self._check_frames_queue())
        
        # Start silence monitoring
        asyncio.create_task(self._monitor_silence())
    
    async def _monitor_silence(self):
        """Monitor for extended silence to end conversation"""
        while self.running:
            current_time = time.time()
            # Only timeout if AI is not speaking, not generating speech, and user is not speaking
            if (not self.is_ai_speaking and 
                not self.is_generating_speech and
                not self.vad.speaking and
                current_time - self.last_activity_time > AudioConfig.CONVERSATION_TIMEOUT):
                print("\nConversation timed out due to silence.")
                self.running = False
                # Signal the main loop to exit
                if self.callback:
                    await self.callback(None, terminate=True)
                break
            await asyncio.sleep(1)
    
    async def _check_frames_queue(self):
        """Check the queue for processed audio frames"""
        while self.running:
            # Check if there are frames to process
            if not self.frames_queue.empty():
                frames = self.frames_queue.get()
                self.last_activity_time = time.time()
                await self.callback(frames)
            await asyncio.sleep(0.1)
    
    def _process_audio(self):
        """Process incoming audio with VAD"""
        try:
            while self.running and not FORCE_STOP:
                # Read audio chunk
                chunk = self.stream.read(AudioConfig.CHUNK, exception_on_overflow=False)
                
                # If AI is speaking and human starts speaking, signal an interruption
                if self.is_ai_speaking:
                    rms = audioop.rms(chunk, 2)
                    if rms > AudioConfig.INTERRUPTION_THRESHOLD:
                        print("\nUser interrupted. Stopping AI response...")
                        self.is_ai_speaking = False
                
                if self.vad.is_speech(chunk):
                    # Visual feedback during recording
                    rms = audioop.rms(chunk, 2)
                    bars = min(40, int(rms / 50))
                    print(f"\r[{'|' * bars}{' ' * (40-bars)}] {rms:5d}", end="", flush=True)
                    
                    # Update activity timestamp
                    self.last_activity_time = time.time()
                else:
                    # If we were speaking but now stopped, process the audio
                    if self.vad.frames:
                        print("\nSpeech detected, processing...")
                        frames = self.vad.reset()
                        
                        # Add frames to the queue for processing by the main thread
                        if frames:
                            self.frames_queue.put(frames)
                    
        except Exception as e:
            print(f"Error in audio processing: {e}")
        finally:
            # Clean up
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
    
    def set_ai_speaking(self, is_speaking: bool):
        """Set whether the AI is currently speaking
        
        Args:
            is_speaking: True if AI is speaking, False otherwise
        """
        self.is_ai_speaking = is_speaking
        if is_speaking:
            self.last_activity_time = time.time()
    
    def set_generating_speech(self, is_generating: bool):
        """Set whether the AI is currently generating speech
        
        Args:
            is_generating: True if generating speech, False otherwise
        """
        self.is_generating_speech = is_generating
        if is_generating:
            self.last_activity_time = time.time()
    
    async def stop(self):
        """Stop listening for audio and clean up resources"""
        self.running = False
        if hasattr(self, 'audio_thread') and self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=1.0)
        await asyncio.sleep(0.1)  # Give time for cleanup
        
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception as e:
                print(f"Error closing audio stream: {e}")
                
        try:
            self.audio.terminate()
        except Exception as e:
            print(f"Error terminating PyAudio: {e}")

# -----------------------------------------------------------------------------
# Speech-to-Text Service
# -----------------------------------------------------------------------------

class OpenAISTTService:
    """Speech-to-Text using OpenAI's Whisper API"""
    
    def __init__(self, model: str = AIConfig.STT_MODEL):
        """Initialize the STT service
        
        Args:
            model: The Whisper model to use
        """
        self.client = AsyncOpenAI()
        self.model = model
    
    async def transcribe(self, audio_frames: List[bytes]) -> str:
        """Transcribe audio frames to text
        
        Args:
            audio_frames: List of audio frames to transcribe
            
        Returns:
            The transcribed text
        """
        # Save frames to a temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            filename = temp_file.name
            
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(AudioConfig.CHANNELS)
            wf.setsampwidth(2)  # 2 bytes for FORMAT=paInt16
            wf.setframerate(AudioConfig.RATE)
            wf.writeframes(b''.join(audio_frames))
        
        try:
            # Transcribe using OpenAI
            with open(filename, "rb") as audio_file:
                transcription = await self.client.audio.transcriptions.create(
                    model=self.model,
                    file=audio_file,
                    language="en"  # Force English language recognition
                )
            
            text = transcription.text.strip()
            print(f"You said: {text}")
            return text
        except Exception as e:
            print(f"Error transcribing: {e}")
            return ""
        finally:
            # Clean up
            if os.path.exists(filename):
                try:
                    os.unlink(filename)
                except Exception as e:
                    print(f"Error removing temporary file: {e}")

# -----------------------------------------------------------------------------
# Language Model Service
# -----------------------------------------------------------------------------

class OpenAILLMService:
    """Large Language Model using OpenAI's API"""
    
    def __init__(self, model: str = AIConfig.LLM_MODEL, 
                 temperature: float = AIConfig.TEMPERATURE):
        """Initialize the LLM service
        
        Args:
            model: The GPT model to use
            temperature: Response temperature (0-1)
        """
        self.client = AsyncOpenAI()
        self.model = model
        self.temperature = temperature
        self.messages = [
            {"role": "system", "content": AIConfig.SYSTEM_PROMPT}
        ]
    
    async def get_response(self, user_input: str) -> str:
        """Get AI response to user input
        
        Args:
            user_input: The user's text input
            
        Returns:
            The AI's text response
        """
        if not user_input.strip():
            return "I didn't catch that. Could you repeat?"
        
        # Add user message
        self.messages.append({"role": "user", "content": user_input})
        
        try:
            # Stream the response for faster first token
            response_text = ""
            print("AI: ", end="", flush=True)
            
            # Create a streaming response
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                temperature=self.temperature,
                stream=True
            )
            
            # Process the streaming response
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    content_piece = chunk.choices[0].delta.content
                    response_text += content_piece
                    print(content_piece, end="", flush=True)
            
            print()  # Add newline after response
            
            # Add response to message history
            self.messages.append({"role": "assistant", "content": response_text})
            
            return response_text
        except Exception as e:
            print(f"Error getting AI response: {e}")
            return "I'm having trouble processing your request right now."

# -----------------------------------------------------------------------------
# Text-to-Speech Service
# -----------------------------------------------------------------------------

class OpenAITTSService:
    """Text-to-Speech using OpenAI's API"""
    
    def __init__(self, model: str = AIConfig.TTS_MODEL, 
                 voice: str = AIConfig.TTS_VOICE):
        """Initialize the TTS service
        
        Args:
            model: The TTS model to use
            voice: The voice to use
        """
        self.client = AsyncOpenAI()
        self.model = model
        self.voice = voice
        self.temp_file = None
        self.process = None
        self.should_interrupt = False
    
    async def synthesize(self, text: str, audio_service: Optional[AudioInputService] = None):
        """Convert text to speech
        
        Args:
            text: The text to convert to speech
            audio_service: Optional AudioInputService to update speaking state
        """
        if not text:
            print("Empty text received, skipping speech synthesis")
            return
        
        # Clean up any previous temp files
        self._cleanup_temp_file()
        self.should_interrupt = False
            
        try:
            print("Generating speech...")
            
            # Signal that AI is starting to generate speech
            if audio_service:
                audio_service.set_generating_speech(True)
                
            # Generate speech
            response = await self.client.audio.speech.create(
                model=self.model,
                voice=self.voice,
                input=text,
                speed=1.1
            )
            
            # Signal that generation is complete
            if audio_service:
                audio_service.set_generating_speech(False)
            
            # Create a temporary file
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
                self.temp_file = temp_file.name
                
            # Write the response content to the file
            with open(self.temp_file, "wb") as f:
                f.write(response.content)
                
            print("Speaking response...")
            
            # Signal that AI is starting to speak
            if audio_service:
                audio_service.set_ai_speaking(True)
            
            # Play the audio with more frequent interrupt checks
            if sys.platform == "darwin":  # macOS
                self.process = await asyncio.create_subprocess_exec(
                    "afplay", self.temp_file,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                # Wait for playback to complete or be interrupted
                while not self.should_interrupt:
                    try:
                        # Check more frequently (every 0.05s instead of 0.1s)
                        done, _ = await asyncio.wait([self.process.wait()], timeout=0.05)
                        if done:  # Process finished
                            break
                    except asyncio.TimeoutError:
                        # Check if we should interrupt
                        if self.should_interrupt:
                            self.process.terminate()
                            break
                        
                        # While talking, still listen for microphone input to check for interruptions
                        if audio_service and audio_service.stream:
                            try:
                                chunk = audio_service.stream.read(AudioConfig.CHUNK, exception_on_overflow=False)
                                rms = audioop.rms(chunk, 2)
                                if rms > AudioConfig.INTERRUPTION_THRESHOLD:
                                    print(f"\nHigh volume detected during speech (RMS: {rms}). Stopping response...")
                                    self.should_interrupt = True
                                    self.process.terminate()
                                    break
                            except Exception as e:
                                pass  # Ignore errors in this check
            
            elif sys.platform == "linux":  # Linux
                os.system(f"aplay {self.temp_file}")
            elif sys.platform == "win32":  # Windows
                os.system(f"start {self.temp_file}")
                
            # Signal that AI is done speaking and reset the activity timer
            if audio_service:
                audio_service.set_ai_speaking(False)
                # Reset activity timer after AI finishes speaking to give full timeout period
                audio_service.last_activity_time = time.time()
                
            # Clean up
            self._cleanup_temp_file()
                
        except Exception as e:
            print(f"Error synthesizing speech: {str(e)}")
            self._cleanup_temp_file()
            
            # Ensure AI speaking and generating states are reset
            if audio_service:
                audio_service.set_ai_speaking(False)
                audio_service.set_generating_speech(False)
                # Reset activity timer even if TTS fails
                audio_service.last_activity_time = time.time()
    
    def interrupt(self):
        """Interrupt current speech playback"""
        self.should_interrupt = True
        if self.process:
            try:
                self.process.terminate()
            except Exception as e:
                print(f"Error terminating TTS process: {e}")
    
    def _cleanup_temp_file(self):
        """Clean up temporary audio file"""
        if self.temp_file and os.path.exists(self.temp_file):
            try:
                os.unlink(self.temp_file)
                self.temp_file = None
            except Exception as e:
                print(f"Error cleaning up temporary file: {e}")

# -----------------------------------------------------------------------------
# Main Pipeline
# -----------------------------------------------------------------------------

class VoiceAgentPipeline:
    """Main voice conversation pipeline"""
    
    def __init__(self):
        """Initialize the voice agent pipeline"""
        # Initialize services
        self.audio_input = AudioInputService()
        self.stt = OpenAISTTService()
        self.llm = OpenAILLMService()
        self.tts = OpenAITTSService()
        
        # Control flags
        self.running = True
        self.stop_requested = False
        
        # Initialize file paths
        self.cwd = os.getcwd()
        self.stop_file_path = os.path.join(self.cwd, "STOP")
        self.force_stop_file_path = os.path.join(self.cwd, "FORCE_STOP")
        
        # Start file watcher
        self.watcher_thread = threading.Thread(target=self._file_watcher)
        self.watcher_thread.daemon = True
    
    def _file_watcher(self):
        """Watch for stop files"""
        global FORCE_STOP
        
        while not self.stop_requested and not FORCE_STOP:
            # Check if a special file exists
            if os.path.exists(self.stop_file_path) or os.path.exists(self.force_stop_file_path):
                print("\nSTOP file detected! Ending conversation...")
                if os.path.exists(self.force_stop_file_path):
                    FORCE_STOP = True
                    os._exit(0)  # Force exit
                self.stop_requested = True
                # Remove the files
                self._remove_stop_files()
            time.sleep(0.5)
    
    def _remove_stop_files(self):
        """Remove any stop files"""
        for file_path in [self.stop_file_path, self.force_stop_file_path]:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    print(f"Removed stop file: {file_path}")
                except Exception as e:
                    print(f"Error removing file {file_path}: {e}")

    # Add a method to create stop files directly from the pipeline
    def request_stop(self, force: bool = False) -> bool:
        """Request the voice agent to stop.
        
        Args:
            force: Whether to force immediate termination
            
        Returns:
            True if the request was made, False otherwise
        """
        if has_utils:
            # Use the utility module if available
            return voice_agent_utils.create_stop_file(force=force)
        else:
            # Direct implementation
            file_path = self.force_stop_file_path if force else self.stop_file_path
            try:
                with open(file_path, 'w') as f:
                    f.write("STOP")
                print(f"Created {'FORCE_STOP' if force else 'STOP'} file")
                return True
            except Exception as e:
                print(f"Error creating stop file: {e}")
                return False

    async def process_speech(self, frames, terminate=False, is_interrupt=False):
        """Process detected speech
        
        Args:
            frames: Audio frames to process
            terminate: Whether this is a termination signal
            is_interrupt: Whether this is an interruption
        """
        # Handle termination signal from silence detector
        if terminate:
            self.running = False
            return
            
        # Handle user interruption
        if self.audio_input.is_ai_speaking:
            self.tts.interrupt()
            self.audio_input.set_ai_speaking(False)
            # Wait a moment to let the interruption complete
            await asyncio.sleep(0.2)
        
        # No frames means this is just a control signal, not actual speech
        if not frames:
            return
        
        # Convert speech to text
        text = await self.stt.transcribe(frames)
        
        # Check for termination keywords
        if TerminationConfig.check_for_termination(text):
            print(f"\nTermination keyword detected: '{text}'")
            print("Ending conversation...")
            self.running = False
            return
        
        if text:
            # Get AI response
            response = await self.llm.get_response(text)
            
            # Convert response to speech
            await self.tts.synthesize(response, self.audio_input)
    
    async def run(self):
        """Run the voice agent pipeline"""
        global FORCE_STOP
        
        # Start file watcher
        self.watcher_thread.start()
        
        print("\nTo stop the program at any time:")
        print("1. Press Ctrl+C")
        print(f"2. Create a file named 'STOP' in: {self.cwd}")
        print(f"3. For emergency termination, create a file named 'FORCE_STOP' in: {self.cwd}")
        
        try:
            # Start listening
            await self.audio_input.start(self.process_speech)
            
            # Keep the program running until termination is requested
            print("Press Ctrl+C or say 'stop', 'exit', 'quit', etc. to end the conversation")
            print(f"The conversation will also end after {AudioConfig.CONVERSATION_TIMEOUT} seconds of silence")
            
            while self.running and not self.stop_requested and not FORCE_STOP:
                await asyncio.sleep(0.5)
                
            print("Conversation ended.")
            
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            # Update running state
            self.running = False
            FORCE_STOP = True
            
            print("Cleaning up resources...")
            
            # Clean up and ensure we exit properly
            await self.audio_input.stop()
            
            # Make sure TTS cleanup is performed
            if self.tts.temp_file:
                self.tts._cleanup_temp_file()
            
            # Make sure we don't leave stop files around
            self._remove_stop_files()
            
            print("All resources cleaned up. Exiting...")

# -----------------------------------------------------------------------------
# Signal Handling
# -----------------------------------------------------------------------------

def signal_handler(sig, frame):
    """Handle Ctrl+C to stop the program immediately
    
    Args:
        sig: Signal number
        frame: Current stack frame
    """
    global FORCE_STOP
    print("\nCtrl+C detected! Force stopping...")
    FORCE_STOP = True

# Register signal handler for Ctrl+C
signal.signal(signal.SIGINT, signal_handler)

# -----------------------------------------------------------------------------
# Entry Point
# -----------------------------------------------------------------------------

async def main():
    """Main entry point"""
    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is required")
        print("Please set it in your .env file or environment variables")
        return 1
    
    # Process command line arguments
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        # Add built-in stop functionality
        if arg in ["stop", "--stop", "-s"]:
            if has_utils:
                force = "--force" in sys.argv or "-f" in sys.argv
                kill = "--kill" in sys.argv or "-k" in sys.argv
                voice_agent_utils.stop_voice_agent(force=force, kill_processes=kill)
            else:
                # Create a stop file directly
                with open("STOP", "w") as f:
                    f.write("STOP")
                print("STOP file created. The voice agent should terminate soon.")
            return 0
        # Add built-in kill functionality
        elif arg in ["kill", "--kill", "-k"]:
            if has_utils:
                voice_agent_utils.kill_python_processes()
            else:
                print("Kill functionality requires voice_agent_utils.py")
            return 0
        
    print("=== OpenAI Voice Conversation Pipeline ===")
    print("Start speaking when ready. The system will detect when you pause.")
    
    pipeline = VoiceAgentPipeline()
    await pipeline.run()
    return 0

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except Exception as e:
        print(f"Error in main pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 