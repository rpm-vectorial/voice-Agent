"""
Streaming Voice Conversation with OpenAI
Continuous listening mode with automatic silence detection
"""

import os
import tempfile
import time
import threading
import queue
import keyboard
import pyaudio
import wave
import numpy as np
import subprocess
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ensure API key is available
if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError("OPENAI_API_KEY environment variable is not set")

# Initialize OpenAI client
client = OpenAI()

# Audio recording settings
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 24000
DEFAULT_VOICE = "nova"
SILENCE_THRESHOLD = 30  # Volume level to detect silence
SILENCE_CHUNKS = 20     # Number of chunks of silence to trigger end of speech
LISTENING_TIMEOUT = 10  # Seconds of silence before going back to standby
MIN_SPEECH_LENGTH = 5   # Minimum frames needed for valid speech detection

# Constants for models from screenshot
GPT_MODEL = "gpt-4o-realtime-preview"
WHISPER_MODEL = "whisper-1"
GPT_TEMPERATURE = 0.8
CONVERSATION_HISTORY = []

# State control
is_listening = False
is_speaking = False
is_processing = False
should_exit = False
audio_queue = queue.Queue()
standby_mode = True

class VoiceRecorder:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.frames = []
        self.is_recording = False
        self.is_passive_listening = True
        self.last_sound_time = time.time()
        self.silence_counter = 0
        self.speech_detected = False
        self.device_index = None
        
    def setup_stream(self, device_index):
        """Setup audio stream with the selected microphone"""
        self.device_index = device_index
        self.stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=CHUNK
        )
        return True
        
    def passive_listening(self):
        """Continuously listen for speech and automatically start recording when detected"""
        global is_listening, standby_mode
        
        print("Passive listening mode... (speak to activate or press 'q' to quit)")
        
        self.is_passive_listening = True
        max_volume = 0
        standby_frames = 0
        
        try:
            while self.is_passive_listening and not should_exit:
                if self.stream is None:
                    self.setup_stream(self.device_index)
                    
                data = self.stream.read(CHUNK, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.int16)
                volume_norm = np.linalg.norm(audio_data) / 100
                
                # Update max volume for display
                max_volume = max(max_volume, volume_norm)
                
                # Visual feedback in standby mode
                if standby_mode:
                    standby_frames += 1
                    if standby_frames % 5 == 0:  # Update less frequently
                        meter = "=" * int(min(volume_norm, 50)) + "_" * (50 - int(min(volume_norm, 50)))
                        print(f"\r👂 Standby [{meter}] {volume_norm:.1f}  ", end="", flush=True)
                
                # If volume is above threshold, start active recording
                if volume_norm > SILENCE_THRESHOLD:
                    print("\rSpeech detected! Recording...                           ")
                    standby_mode = False
                    is_listening = True
                    self.is_passive_listening = False
                    self.start_active_recording()
                    break
                    
                # Small sleep to reduce CPU usage
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("\nStopping passive listening")
        except Exception as e:
            print(f"\nError in passive listening: {e}")
            
    def start_active_recording(self):
        """Actively record speech until silence is detected"""
        global is_processing
        
        print("Recording started... (silence will end recording)")
        self.frames = []
        self.is_recording = True
        self.speech_detected = False
        self.silence_counter = 0
        self.last_sound_time = time.time()
        start_time = time.time()
        
        try:
            while self.is_recording and not should_exit:
                data = self.stream.read(CHUNK, exception_on_overflow=False)
                self.frames.append(data)
                
                # Analyze volume
                audio_data = np.frombuffer(data, dtype=np.int16)
                volume_norm = np.linalg.norm(audio_data) / 100
                
                # Visual feedback
                meter = "=" * int(min(volume_norm, 50)) + "_" * (50 - int(min(volume_norm, 50)))
                rec_time = time.time() - start_time
                print(f"\r🎤 Recording [{meter}] {volume_norm:.1f} ({rec_time:.1f}s)", end="", flush=True)
                
                # Check for speech
                if volume_norm > SILENCE_THRESHOLD:
                    self.speech_detected = True
                    self.last_sound_time = time.time()
                    self.silence_counter = 0
                else:
                    # Count consecutive silence chunks
                    self.silence_counter += 1
                
                # End recording if silence is long enough and we have detected speech
                if self.speech_detected and self.silence_counter >= SILENCE_CHUNKS:
                    print("\nSilence detected, processing speech...")
                    break
                    
                # End if listening timeout exceeded
                if time.time() - self.last_sound_time > LISTENING_TIMEOUT:
                    print("\nListening timeout, ending recording")
                    break
                
                # Small sleep to reduce CPU usage
                time.sleep(0.01)
                
            # Process the recording
            is_processing = True
            self.is_recording = False
            
            # Check if we have enough frames and detected speech
            if len(self.frames) > MIN_SPEECH_LENGTH and self.speech_detected:
                audio_file = self.save_recording()
                if audio_file:
                    # Add to processing queue
                    audio_queue.put(audio_file)
                    return True
            else:
                print("Not enough speech detected, going back to standby mode")
                
            # Return to passive listening
            self.return_to_passive()
            return False
                
        except KeyboardInterrupt:
            print("\nRecording stopped manually")
            self.is_recording = False
        except Exception as e:
            print(f"\nError in active recording: {e}")
            self.is_recording = False
        
        return False
        
    def save_recording(self):
        """Save the recorded frames to a file"""
        if not self.frames:
            return None
            
        try:
            # Create a temporary file
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_filename = temp_file.name
            temp_file.close()
            
            # Save recording to WAV file
            with wave.open(temp_filename, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(self.audio.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(self.frames))
                
            print(f"Recording saved: {len(self.frames)} frames, {os.path.getsize(temp_filename)} bytes")
            return temp_filename
            
        except Exception as e:
            print(f"Error saving recording: {e}")
            return None
            
    def return_to_passive(self):
        """Return to passive listening mode"""
        global is_listening, standby_mode
        
        self.frames = []
        self.is_recording = False
        is_listening = False
        standby_mode = True
        
        # Start passive listening in the background
        passive_thread = threading.Thread(target=self.passive_listening)
        passive_thread.daemon = True
        passive_thread.start()
        
    def cleanup(self):
        """Clean up resources"""
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except:
                pass
                
        try:
            self.audio.terminate()
        except:
            pass

def list_microphones():
    """List available microphones and let user select one"""
    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    
    print("\nAvailable microphones:")
    microphones = []
    device_names = []
    
    for i in range(numdevices):
        try:
            device_info = p.get_device_info_by_host_api_device_index(0, i)
            if device_info and device_info.get('maxInputChannels') > 0:
                print(f"{len(microphones)}: {device_info.get('name')} (Device {i})")
                microphones.append(i)
                device_names.append(device_info.get('name'))
        except:
            pass
    
    if not microphones:
        print("No microphones found!")
        p.terminate()
        return None
    
    selection = input("\nSelect microphone number (or press Enter for default): ")
    p.terminate()
    
    if selection.strip() and selection.isdigit() and 0 <= int(selection) < len(microphones):
        selected_idx = microphones[int(selection)]
        print(f"Using microphone: {device_names[int(selection)]}")
        return selected_idx
    
    print(f"Using default microphone: {device_names[0]}")
    return microphones[0]

def transcribe_audio(audio_file_path):
    """Transcribe audio using OpenAI's Whisper model"""
    try:
        print(f"Transcribing audio...")
        
        with open(audio_file_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model=WHISPER_MODEL,
                file=audio_file
            )
        
        if not transcription or not transcription.text:
            return ""
        
        result = transcription.text.strip()
        print(f"You said: {result}")
        return result
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return ""

def get_ai_response(user_input):
    """Get AI response using the model from screenshot with conversation history"""
    global CONVERSATION_HISTORY
    
    if not user_input:
        return "I couldn't understand that. Could you please repeat?"
    
    print("Generating response...")
    
    try:
        # Prepare messages with conversation history
        messages = [
            {
                "role": "system", 
                "content": "You are a helpful, friendly voice assistant. Keep responses natural and concise."
            }
        ]
        
        # Add conversation history
        for entry in CONVERSATION_HISTORY[-10:]:
            messages.append(entry)
            
        # Add current user input
        messages.append({"role": "user", "content": user_input})
        
        # Use the model from the screenshot with fallback
        try:
            response = client.chat.completions.create(
                model=GPT_MODEL,
                temperature=GPT_TEMPERATURE,
                messages=messages
            )
        except Exception:
            # Fallback to reliable model
            response = client.chat.completions.create(
                model="gpt-4o",
                temperature=GPT_TEMPERATURE,
                messages=messages
            )
        
        ai_response = response.choices[0].message.content
        
        # Save to conversation history
        CONVERSATION_HISTORY.append({"role": "user", "content": user_input})
        CONVERSATION_HISTORY.append({"role": "assistant", "content": ai_response})
        
        print(f"Assistant: {ai_response}")
        return ai_response
    except Exception as e:
        print(f"Error generating response: {e}")
        return "I'm sorry, I couldn't process your request right now."

def speak_response(text, voice=DEFAULT_VOICE):
    """Convert text to speech using OpenAI TTS"""
    global is_speaking
    
    if not text:
        return
        
    is_speaking = True
    print("Speaking response...")
    
    try:
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        temp_filename = temp_file.name
        temp_file.close()
        
        # Generate speech
        speech_response = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text,
            speed=1.1
        )
        
        # Save to file
        with open(temp_filename, "wb") as f:
            f.write(speech_response.content)
        
        # Play using system command
        if os.name == 'posix':
            subprocess.run(['afplay' if 'darwin' in os.uname().sysname.lower() else 'aplay', temp_filename], 
                         check=True)
        elif os.name == 'nt':
            os.startfile(temp_filename)
        
        # Clean up after playback
        time.sleep(0.5)
        if os.path.exists(temp_filename):
            os.unlink(temp_filename)
            
    except Exception as e:
        print(f"Error speaking response: {e}")
        
    is_speaking = False

def process_audio_files():
    """Process audio files from the queue"""
    global is_processing, standby_mode
    
    while not should_exit:
        try:
            if not audio_queue.empty():
                audio_file = audio_queue.get()
                is_processing = True
                standby_mode = False
                
                # Process the audio file
                transcript = transcribe_audio(audio_file)
                
                # Only continue if we got text
                if transcript:
                    # Get AI response
                    response = get_ai_response(transcript)
                    
                    # Speak the response
                    speak_response(response)
                    
                # Clean up the file
                if os.path.exists(audio_file):
                    os.unlink(audio_file)
                    
                # Reset processing state
                is_processing = False
                standby_mode = True
            
            # Small sleep to reduce CPU usage
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Error processing audio: {e}")
            is_processing = False
            standby_mode = True
            
def key_listener():
    """Listen for keyboard events"""
    global should_exit
    
    print("Press 'q' to quit...")
    
    while not should_exit:
        try:
            if keyboard.is_pressed('q'):
                print("\nExiting...")
                should_exit = True
                break
                
            # Small sleep to reduce CPU usage
            time.sleep(0.1)
            
        except:
            pass

def main():
    """Main function for streaming voice conversation"""
    global should_exit
    
    print("=== Streaming Voice Conversation System ===")
    print("Just speak naturally, and I'll respond")
    print("The system automatically detects when you pause")
    print("Voice commands:")
    print("  'exit' or 'quit' - End the program")
    print("  'clear history' - Reset conversation")
    
    # Select microphone
    mic_index = list_microphones()
    if mic_index is None:
        print("No microphone available. Exiting.")
        return
    
    recorder = VoiceRecorder()
    recorder.setup_stream(mic_index)
    
    try:
        # Start the audio processor thread
        processor_thread = threading.Thread(target=process_audio_files)
        processor_thread.daemon = True
        processor_thread.start()
        
        # Start the key listener thread
        key_thread = threading.Thread(target=key_listener)
        key_thread.daemon = True
        key_thread.start()
        
        # Start in passive listening mode
        recorder.passive_listening()
        
        # Main loop
        while not should_exit:
            # If not listening, processing, or speaking, go to passive mode
            if not is_listening and not is_processing and not is_speaking:
                recorder.return_to_passive()
                
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error in main loop: {e}")
    finally:
        should_exit = True
        recorder.cleanup()

if __name__ == "__main__":
    main() 