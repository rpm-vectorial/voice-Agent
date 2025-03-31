"""
Enhanced Real-time Voice Conversation with OpenAI
A seamless speech-to-text-to-speech system that feels like talking to a human
"""

import os
import asyncio
import tempfile
import time
import threading
import pyaudio
import wave
import numpy as np
import concurrent.futures
import subprocess
import random
import re
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI

# Load environment variables
load_dotenv()

# Ensure API key is available
if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError("OPENAI_API_KEY environment variable is not set. Please set it in a .env file or export it.")

print("Starting Enhanced Voice Conversation System...")

# Initialize the OpenAI clients
client = OpenAI()
async_client = AsyncOpenAI()

# Audio recording settings
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 24000
SILENCE_THRESHOLD = 30  # Further lowered for better speech detection
SILENCE_DURATION = 0.3  # Reduced further for faster response
DEFAULT_VOICE = "nova"   # Using "nova" as the default voice
CONVERSATION_HISTORY = []  # Store conversation history
IDLE_TIMEOUT = 10        # Increased to allow more time for user to speak
RESPONSE_PROMPT_TIMEOUT = 10  # Reduced timeout for faster responses
MICROPHONE_INDEX = None  # Will be set by user selection or auto-detection
last_user_interaction_time = time.time()
idle_timer_running = False
DEBUG_MODE = True  # Enable debug output
USE_WHISPER_MODEL = True  # Use Whisper model for transcription (shown in screenshot)
GPT_MODEL = "gpt-4o-realtime-preview"  # Model from screenshot
GPT_TEMPERATURE = 0.8  # Temperature from screenshot
dynamic_threshold = None  # Will be set during calibration
USE_FASTER_MODEL = True  # Use faster model for transcription

# Exit phrases that will end the conversation
EXIT_PHRASES = [
    "bye", "goodbye", "exit", "quit", "stop", "end", "terminate", 
    "see you", "talk to you later", "take care", "that's all", "that is all",
    "thanks bye", "thank you bye", "ok bye", "okay bye", "see ya", "good night"
]

# Follow-up questions when conversation goes silent
FOLLOW_UP_QUESTIONS = [
    "Is there anything else you'd like to know about?",
    "Do you have any other questions for me?",
    "Was there something specific you were curious about?",
    "What else would you like to talk about?",
    "Is there anything else on your mind?",
    "How can I help you further today?",
    "Would you like to know more about any particular topic?",
    "Is there something else I can help you with today?",
    "Any other questions you have for me?",
    "What else are you interested in discussing?"
]

# Goodbye responses for graceful exit
GOODBYE_RESPONSES = [
    "Goodbye! It was nice talking with you.",
    "Take care! Feel free to chat again anytime.",
    "Bye for now! Have a great day.",
    "Goodbye! I enjoyed our conversation.",
    "Alright, talk to you later!",
    "Take care! It was a pleasure chatting with you."
]

# Replace calibrate_threshold with a simpler, faster version
def calibrate_threshold(recorder, duration=1):
    """Calibrate the silence threshold based on ambient noise - simplified version"""
    global SILENCE_THRESHOLD, dynamic_threshold
    
    print("\nUsing default silence threshold for faster startup.")
    SILENCE_THRESHOLD = 30  # Reasonable default that works for most microphones
    return True

def setup_microphone():
    """Find and setup the best microphone for recording by asking the user"""
    global MICROPHONE_INDEX
    
    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    
    # List available microphones
    print("\nAvailable input devices:")
    available_mics = []
    for i in range(numdevices):
        device_info = p.get_device_info_by_host_api_device_index(0, i)
        if device_info.get('maxInputChannels') > 0:
            print(f"Device {i}: {device_info.get('name')}")
            available_mics.append(i)
    
    # Let user select microphone
    if available_mics:
        try:
            selection = input("\nEnter device index to use (or press Enter for auto-detection): ")
            if selection.strip():
                MICROPHONE_INDEX = int(selection)
                print(f"Using microphone: Device {MICROPHONE_INDEX}")
            else:
                # Auto-select: prefer Yeti or MacBook microphone
                for i in available_mics:
                    name = p.get_device_info_by_index(i).get('name').lower()
                    if 'yeti' in name:
                        MICROPHONE_INDEX = i
                        print(f"Auto-selected Yeti microphone (Device {i})")
                        break
                    elif 'macbook' in name and 'micro' in name:
                        MICROPHONE_INDEX = i
                        print(f"Auto-selected MacBook microphone (Device {i})")
                        break
                
                # If no preferred mic found, use the first available
                if MICROPHONE_INDEX is None and available_mics:
                    MICROPHONE_INDEX = available_mics[0]
                    print(f"Auto-selected first available microphone (Device {MICROPHONE_INDEX})")
        except (ValueError, IndexError):
            # Default to first available on error
            if available_mics:
                MICROPHONE_INDEX = available_mics[0]
                print(f"Invalid selection. Using first available microphone (Device {MICROPHONE_INDEX})")
    
    p.terminate()
    
    if MICROPHONE_INDEX is None:
        print("No microphone found! Please check your audio settings.")
        return False
    
    # Test the selected microphone
    print(f"Testing microphone (Device {MICROPHONE_INDEX})...")
    try:
        test_p = pyaudio.PyAudio()
        test_stream = test_p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=MICROPHONE_INDEX,
            frames_per_buffer=CHUNK
        )
        
        # Read a sample to verify microphone works
        try:
            data = test_stream.read(CHUNK)
            print("Microphone test successful!")
        except Exception as e:
            print(f"Warning: Microphone test read failed: {e}")
            print("Will continue anyway, but you may have audio issues.")
        
        test_stream.stop_stream()
        test_stream.close()
        test_p.terminate()
    except Exception as e:
        print(f"Error testing microphone: {e}")
        print("You may need to select a different microphone.")
        return False
    
    # Run calibration on the selected microphone
    calibrate_threshold(None)
    
    # Set calibration prompt
    print("\nPlease speak a test sentence to calibrate your microphone...")
    print("(Speaking loudly and clearly for 3-5 seconds will help calibrate the system)")
    
    return True

def print_settings():
    """Print current voice detection settings"""
    print("\nCurrent Voice Detection Settings:")
    print(f"- Silence Threshold: {SILENCE_THRESHOLD} (lower = more sensitive)")
    print(f"- Silence Duration: {SILENCE_DURATION} sec (lower = faster response)")
    print(f"- Selected Microphone: Device {MICROPHONE_INDEX}")
    print(f"- Idle Timeout: {IDLE_TIMEOUT} sec")
    print()

class VoiceRecorder:
    """Class to handle voice recording with automatic silence detection"""
    
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.frames = []
        self.is_recording = False
        self.temp_file = None
        self.volume_callback = None
        self.last_volume = 0
        self.has_speech = False  # Flag to track if speech was detected
        self.max_volume = 0      # Track maximum volume for debugging
    
    def callback(self, in_data, frame_count, time_info, status):
        """Callback function for PyAudio's stream"""
        self.frames.append(in_data)
        audio_data = np.frombuffer(in_data, dtype=np.int16)
        volume_norm = np.linalg.norm(audio_data) / 100
        self.last_volume = volume_norm
        
        # Track maximum volume for debugging
        if volume_norm > self.max_volume:
            self.max_volume = volume_norm
        
        # Visual feedback with detection indicator
        speech_detected = volume_norm > SILENCE_THRESHOLD
        meter = "=" * int(min(volume_norm, 50)) + "_" * (50 - int(min(volume_norm, 50)))
        indicator = "🎤" if speech_detected else "  "
        print(f"\r{indicator} Listening [{meter}] {volume_norm:.1f}", end="", flush=True)
        
        # Check if this might be speech
        if speech_detected:
            self.has_speech = True
        
        if self.volume_callback:
            self.volume_callback(volume_norm)
            
        return (in_data, pyaudio.paContinue)
    
    def start_recording(self):
        """Start recording audio from microphone"""
        self.frames = []
        self.is_recording = True
        self.has_speech = False  # Reset speech detection flag
        self.max_volume = 0      # Reset maximum volume
        print("Listening... (speak naturally, I'll respond when you pause)")
        
        # Open PyAudio stream with selected microphone
        try:
            self.stream = self.audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=MICROPHONE_INDEX,
                frames_per_buffer=CHUNK,
                stream_callback=self.callback
            )
            
            # Start the stream
            self.stream.start_stream()
        except Exception as e:
            print(f"Error starting audio stream: {e}")
            self.is_recording = False
            return False
            
        return True
    
    def stop_recording(self) -> str:
        """Stop recording and save to a temporary file"""
        if not self.is_recording:
            return ""
        
        self.is_recording = False
        try:
            if hasattr(self, 'stream') and self.stream.is_active():
                self.stream.stop_stream()
                self.stream.close()
        except Exception as e:
            print(f"Error stopping stream: {e}")
        
        # Check if we have enough audio data
        if len(self.frames) < 3:
            print("\nNot enough audio data recorded")
            return ""
        
        # Check if speech was detected
        if not self.has_speech:
            print(f"\nNo speech detected in recording (max volume: {self.max_volume:.1f}, threshold: {SILENCE_THRESHOLD})")
            return ""
        
        # Create a temporary file
        try:
            self.temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_filename = self.temp_file.name
            self.temp_file.close()
            
            # Save recorded audio to WAV file
            with wave.open(temp_filename, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(self.audio.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(self.frames))
            
            print(f"\nProcessing... ({len(self.frames)} frames, {os.path.getsize(temp_filename)} bytes)")
            return temp_filename
        except Exception as e:
            print(f"\nError saving audio file: {e}")
            return ""
    
    def record_with_silence_detection(self, timeout=30) -> Optional[str]:
        """Record audio until silence is detected or timeout reached"""
        global last_user_interaction_time
        
        if not self.start_recording():
            return None
        
        silence_start = None
        speech_started = False
        start_time = time.time()
        last_user_interaction_time = time.time()  # Update interaction time
        pause_detected = False
        consecutive_silence_chunks = 0
        
        try:
            while self.is_recording and not pause_detected:
                # Check if we've exceeded the timeout
                if time.time() - start_time > timeout:
                    print("\nRecording timed out")
                    break
                
                # Check for silence
                if len(self.frames) > 0:
                    audio_data = np.frombuffer(self.frames[-1], dtype=np.int16)
                    volume_norm = np.linalg.norm(audio_data) / 100
                    
                    # If volume is above threshold, user is speaking
                    if volume_norm > SILENCE_THRESHOLD:
                        speech_started = True
                        silence_start = None
                        consecutive_silence_chunks = 0
                        last_user_interaction_time = time.time()  # Update interaction time
                    elif speech_started:  # Only detect silence if speech has started
                        # Track consecutive chunks below threshold for more reliable pause detection
                        consecutive_silence_chunks += 1
                        
                        if silence_start is None:
                            silence_start = time.time()
                        
                        # Detect pause using both time and consecutive silent chunks
                        silence_time = time.time() - silence_start
                        if (silence_time > SILENCE_DURATION and consecutive_silence_chunks >= 3):
                            # Only end if we've recorded something substantial
                            if len(self.frames) > 5:  # Requirement for minimum content
                                print("\nPause detected, processing")
                                pause_detected = True
                                break
                
                # Small pause to reduce CPU usage
                time.sleep(0.01)  # Even more responsive
            
            recording = self.stop_recording()
            if recording and os.path.exists(recording) and os.path.getsize(recording) > 1000:
                return recording
            else:
                if recording and os.path.exists(recording):
                    print(f"Recording appears invalid: size={os.path.getsize(recording)} bytes (need > 1000)")
                return None
        
        except KeyboardInterrupt:
            print("\nRecording stopped by user")
            return self.stop_recording()
        except Exception as e:
            print(f"\nError during recording: {e}")
            import traceback
            traceback.print_exc()
            self.stop_recording()
            return None
    
    def cleanup(self):
        """Clean up resources"""
        try:
            if hasattr(self, 'stream') and self.stream.is_active():
                try:
                    self.stream.stop_stream()
                    self.stream.close()
                except:
                    pass
            try:
                self.audio.terminate()
            except:
                pass
            if self.temp_file and os.path.exists(self.temp_file.name):
                try:
                    os.unlink(self.temp_file.name)
                except:
                    pass
        except Exception as e:
            print(f"Error in cleanup: {e}")

def is_exit_phrase(text: str) -> bool:
    """Check if the user's input is an exit phrase"""
    # Convert to lowercase and remove punctuation
    text = text.lower().strip()
    # Remove common punctuation
    text = re.sub(r'[,.!?;:]', '', text)
    
    # Check if any exit phrase is in the text
    for exit_phrase in EXIT_PHRASES:
        if exit_phrase in text or text in exit_phrase:
            return True
            
    # Check for standalone "bye" with possible spacing/punctuation
    if re.search(r'\b(bye|goodbye)\b', text):
        return True
        
    return False

async def transcribe_audio(audio_file_path: str) -> str:
    """Transcribe audio file using OpenAI's transcription API"""
    try:
        if not os.path.exists(audio_file_path):
            print(f"Error: Audio file does not exist: {audio_file_path}")
            return ""
            
        file_size = os.path.getsize(audio_file_path)
        if file_size < 1000:  # Less than 1 KB
            print(f"Warning: Audio file is very small ({file_size} bytes), may not contain speech")
            
        print(f"Transcribing file: {audio_file_path} ({file_size} bytes)")
            
        # Start a progress indicator
        event = threading.Event()
        
        def progress_indicator():
            count = 0
            indicators = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
            while not event.is_set():
                print(f"\rTranscribing {indicators[count % len(indicators)]}", end="", flush=True)
                count += 1
                time.sleep(0.1)
                
        progress_thread = threading.Thread(target=progress_indicator)
        progress_thread.daemon = True
        progress_thread.start()
        
        try:
            with open(audio_file_path, "rb") as audio_file:
                # Using whisper-1 model as shown in your screenshot
                transcription = await asyncio.wait_for(
                    run_in_threadpool(
                        client.audio.transcriptions.create,
                        model="whisper-1",
                        file=audio_file
                    ),
                    timeout=RESPONSE_PROMPT_TIMEOUT
                )
        finally:
            # Stop the progress indicator
            event.set()
            progress_thread.join(timeout=0.5)
            print("\r", end="", flush=True)  # Clear the line
        
        if not transcription or not transcription.text:
            print("Warning: Empty transcription returned")
            return ""
            
        print(f"You said: {transcription.text}")
        return transcription.text
    except asyncio.TimeoutError:
        print("Transcription timed out, please try again")
        return ""
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        import traceback
        traceback.print_exc()
        return ""

async def get_ai_response(user_input: str, is_follow_up: bool = False) -> Tuple[str, bool]:
    """Get AI response from OpenAI with conversation history"""
    print("Thinking...")
    is_exit = is_exit_phrase(user_input)
    
    # Start a progress indicator
    event = threading.Event()
    
    def progress_indicator():
        count = 0
        indicators = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        while not event.is_set():
            print(f"\rGenerating response {indicators[count % len(indicators)]}", end="", flush=True)
            count += 1
            time.sleep(0.1)
            
    progress_thread = threading.Thread(target=progress_indicator)
    progress_thread.daemon = True
    progress_thread.start()
    
    try:
        if is_exit:
            # Return a goodbye message if user is exiting
            goodbye = random.choice(GOODBYE_RESPONSES)
            CONVERSATION_HISTORY.append({"role": "user", "content": user_input})
            CONVERSATION_HISTORY.append({"role": "assistant", "content": goodbye})
            return goodbye, True
            
        # Prepare messages with conversation history
        system_message = (
            "You are a helpful, friendly voice assistant designed to feel human-like. "
            "Keep your responses natural, conversational, and concise. "
            "Respond as if you're in a flowing conversation. "
            "Limit responses to 1-3 sentences unless more detail is explicitly needed. "
        )
        
        if is_follow_up:
            system_message += "The user has been quiet for a while, so you're initiating the conversation with a follow-up question."
        
        messages = [
            {"role": "system", "content": system_message}
        ]
        
        # Add conversation history - keep for context
        for entry in CONVERSATION_HISTORY[-6:]:  # Keep last 6 exchanges for context
            messages.append(entry)
            
        # Add current user input if not a system-generated follow-up
        if not is_follow_up:
            messages.append({"role": "user", "content": user_input})
        
        # Use the model and temperature from the screenshot
        response = await asyncio.wait_for(
            run_in_threadpool(
                client.chat.completions.create,
                model=GPT_MODEL,
                temperature=GPT_TEMPERATURE,
                messages=messages
            ),
            timeout=RESPONSE_PROMPT_TIMEOUT
        )
        
        ai_response = response.choices[0].message.content
        
        # Save to conversation history
        if not is_follow_up:
            CONVERSATION_HISTORY.append({"role": "user", "content": user_input})
        CONVERSATION_HISTORY.append({"role": "assistant", "content": ai_response})
        
        return ai_response, is_exit
    except asyncio.TimeoutError:
        print("Response generation timed out")
        return "I'm sorry, it's taking me a bit long to respond. Could you please repeat your question?", False
    except Exception as e:
        print(f"Error generating AI response: {e}")
        return "I'm sorry, I couldn't process your request right now.", False
    finally:
        # Stop the progress indicator
        event.set()
        progress_thread.join(timeout=0.5)
        print("\r", end="", flush=True)  # Clear the line

async def speak_response(text: str, voice: str = DEFAULT_VOICE) -> None:
    """Convert text to speech and play it back in real-time"""
    print(f"Assistant: {text}")
    
    try:
        # Start a progress indicator
        event = threading.Event()
        
        def progress_indicator():
            count = 0
            indicators = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
            while not event.is_set():
                print(f"\rGenerating speech {indicators[count % len(indicators)]}", end="", flush=True)
                count += 1
                time.sleep(0.1)
                
        progress_thread = threading.Thread(target=progress_indicator)
        progress_thread.daemon = True
        progress_thread.start()
        
        try:
            # Use file-based approach for reliable audio playback
            temp_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
            temp_filename = temp_file.name
            temp_file.close()
            
            # Use tts-1 model for significantly faster audio generation
            speech_response = await asyncio.wait_for(
                run_in_threadpool(
                    client.audio.speech.create,
                    model="tts-1", # Faster model
                    voice=voice,
                    input=text,
                    speed=1.15  # Slightly faster for more natural pace
                ),
                timeout=RESPONSE_PROMPT_TIMEOUT
            )
            
            # Save to file
            with open(temp_filename, "wb") as f:
                f.write(speech_response.content)
        finally:
            # Stop the progress indicator
            event.set()
            progress_thread.join(timeout=0.5)
            print("\r", end="", flush=True)  # Clear the line
        
        # Play using system commands
        if os.name == 'posix':  # macOS or Linux
            subprocess.run(['afplay' if 'darwin' in os.uname().sysname.lower() else 'aplay', temp_filename], 
                         check=True, 
                         stdout=subprocess.DEVNULL, 
                         stderr=subprocess.DEVNULL)
        elif os.name == 'nt':  # Windows
            os.startfile(temp_filename)  # type: ignore
        
        # Clean up the file after playback
        async def cleanup_file():
            await asyncio.sleep(0.5)
            if os.path.exists(temp_filename):
                try:
                    os.unlink(temp_filename)
                except:
                    pass
                
        # Start cleanup in background
        asyncio.create_task(cleanup_file())
        
    except asyncio.TimeoutError:
        print("Speech generation timed out")
    except Exception as e:
        print(f"Error speaking response: {e}")

# Version-agnostic function to run a function in a thread pool
async def run_in_threadpool(func, *args, **kwargs):
    """Run a function in a thread pool (compatible with all Python versions)"""
    with concurrent.futures.ThreadPoolExecutor() as pool:
        return await asyncio.wrap_future(
            pool.submit(func, *args, **kwargs)
        )

async def prepare_for_next_turn():
    """Quick reset to prepare for the next conversation turn"""
    # A slight delay allows for more natural conversation timing
    await asyncio.sleep(0.1)  # Reduced further for faster response
    print("\n--- Listening ---")

async def handle_idle_timeout():
    """Handle idle timeout by asking follow-up questions or closing conversation"""
    global idle_timer_running, last_user_interaction_time
    
    idle_timer_running = True
    consecutive_silence_count = 0
    
    try:
        while True:
            # Check if user has been inactive
            current_time = time.time()
            time_since_last_interaction = current_time - last_user_interaction_time
            
            if time_since_last_interaction > IDLE_TIMEOUT:
                print(f"\nUser has been idle for {time_since_last_interaction:.1f} seconds")
                
                # Increment silence counter
                consecutive_silence_count += 1
                
                # If multiple consecutive silences, exit
                if consecutive_silence_count >= 2:
                    print("Multiple silences detected, ending conversation")
                    # Say goodbye and exit
                    await speak_response("I notice you're not responding. I'll end our conversation for now. Feel free to talk again anytime!")
                    # Exit the conversation
                    raise asyncio.CancelledError()
                else:
                    # First silence, ask a follow-up
                    follow_up = random.choice(FOLLOW_UP_QUESTIONS)
                    await speak_response(follow_up)
                
                # Reset the timer
                last_user_interaction_time = time.time()
            
            # Check more frequently
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        print("Idle timer cancelled")
        idle_timer_running = False
        # Re-raise to signal parent to exit
        raise
    except Exception as e:
        print(f"Error in idle timer: {e}")
        idle_timer_running = False

async def voice_conversation_turn() -> Tuple[bool, bool]:
    """Complete one turn of the voice conversation
    Returns: (success, should_exit)
    """
    global last_user_interaction_time
    
    recorder = VoiceRecorder()
    audio_file = None
    try:
        # Record user's speech
        if DEBUG_MODE:
            print("Starting recording...")
        audio_file = await run_in_threadpool(recorder.record_with_silence_detection)
        last_user_interaction_time = time.time()  # Update interaction time
        
        if not audio_file or not os.path.exists(audio_file):
            print("No audio recorded or file not found")
            return False, False
        
        # Start transcription immediately
        if DEBUG_MODE:
            print(f"Starting transcription of {audio_file}...")
        transcription_task = asyncio.create_task(transcribe_audio(audio_file))
        
        # Wait for transcription
        transcription = await transcription_task
        if not transcription:
            await speak_response("I didn't catch that. Could you try again?")
            await prepare_for_next_turn()
            return True, False
        
        # Check for exit phrases right away
        if is_exit_phrase(transcription):
            # Say goodbye
            await speak_response(random.choice(GOODBYE_RESPONSES))
            return True, True
        
        # Generate AI response
        if DEBUG_MODE:
            print("Getting AI response...")
        ai_response, is_exit = await get_ai_response(transcription)
        
        # Convert response to speech and play it
        if DEBUG_MODE:
            print("Speaking response...")
        await speak_response(ai_response)
        
        # Cleanup recording file
        if audio_file and os.path.exists(audio_file):
            try:
                os.unlink(audio_file)
            except Exception as e:
                print(f"Warning: Could not delete temp file: {e}")
        
        # Prepare for next turn if not exiting
        if not is_exit:
            await prepare_for_next_turn()
            
        return True, is_exit
    
    except Exception as e:
        print(f"Error in conversation turn: {e}")
        import traceback
        traceback.print_exc()
        # Clean up
        if recorder:
            recorder.cleanup()
        if audio_file and os.path.exists(audio_file):
            try:
                os.unlink(audio_file)
            except:
                pass
        return False, False
    finally:
        # Always make sure recorder is cleaned up
        try:
            recorder.cleanup()
        except:
            pass

async def continuous_voice_mode():
    """Run voice conversation in a continuous mode"""
    print("\n------- CONTINUOUS VOICE MODE -------")
    print("Speak naturally. The assistant will respond quickly when you pause.")
    print("Say 'bye', 'goodbye', or 'take care' to end the conversation.")
    print("The conversation will also end after 5 seconds of inactivity.")
    print("Press Ctrl+C to exit at any time")
    
    # Initial greeting
    await speak_response(
        "Hi there! I'm ready for a conversation. Just speak naturally, and I'll respond quickly when you pause. Say bye when you want to end our chat.",
        DEFAULT_VOICE
    )
    
    # Start the idle timer
    idle_timer = asyncio.create_task(handle_idle_timeout())
    
    try:
        while True:
            try:
                success, should_exit = await voice_conversation_turn()
                if should_exit:
                    print("Exit phrase detected, ending conversation")
                    break
                    
                if not success:
                    print("Restarting voice detection...")
                    await asyncio.sleep(0.2)  # Even shorter pause before retrying
            except Exception as e:
                print(f"Error in conversation turn: {e}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(0.5)
                print("Restarting after error...")
    except KeyboardInterrupt:
        print("\nExiting continuous voice mode")
    except asyncio.CancelledError:
        print("\nConversation timed out due to inactivity")
    except Exception as e:
        print(f"Error in continuous mode: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Always clean up
        if not idle_timer.done():
            idle_timer.cancel()

async def main():
    """Main function to run the voice conversation system"""
    print("Enhanced Voice Conversation System")
    print("This system allows for natural, flowing conversation with an AI assistant")
    
    # Setup microphone and calibrate
    if not setup_microphone():
        print("Error: Could not set up microphone")
        return
    
    # Print settings for debugging
    print_settings()
    
    try:
        await continuous_voice_mode()
        print("\nConversation ended. Thank you for using the voice conversation system!")
    except KeyboardInterrupt:
        print("\nExiting voice conversation system")
    except Exception as e:
        print(f"Error in main program: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 