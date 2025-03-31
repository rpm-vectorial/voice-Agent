"""
Simple Voice Conversation with OpenAI - Manual Recording Version
Press ENTER to start recording, press ENTER again to stop
"""

import os
import asyncio
import tempfile
import time
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
    raise EnvironmentError("OPENAI_API_KEY environment variable is not set. Please set it in a .env file or export it.")

# Initialize OpenAI client
client = OpenAI()

# Audio recording settings
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 24000
DEFAULT_VOICE = "nova"  # Using "nova" as the default voice

# Add constants for models at the top
GPT_MODEL = "gpt-4o-realtime-preview"  # From screenshot
WHISPER_MODEL = "whisper-1"  # From screenshot
GPT_TEMPERATURE = 0.8  # From screenshot
CONVERSATION_HISTORY = []  # Store conversation history

class Recorder:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.frames = []
        self.is_recording = False
        self.recorded_file = None
    
    def start_recording(self, device_index=None):
        """Start recording audio from microphone"""
        self.frames = []
        self.is_recording = True
        
        self.stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=CHUNK
        )
        
        print("Recording started... Press ENTER to stop.")
        
        # Continuous recording in a loop
        while self.is_recording:
            try:
                data = self.stream.read(CHUNK)
                self.frames.append(data)
                
                # Visualize audio levels in real-time
                audio_data = np.frombuffer(data, dtype=np.int16)
                volume_norm = np.linalg.norm(audio_data) / 100
                
                meter = "=" * int(min(volume_norm, 50)) + "_" * (50 - int(min(volume_norm, 50)))
                print(f"\r🎤 Recording [{meter}] {volume_norm:.1f}", end="", flush=True)
            except KeyboardInterrupt:
                break
    
    def stop_recording(self):
        """Stop recording and save to a temporary file"""
        self.is_recording = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        if len(self.frames) < 3:
            print("\nRecording too short")
            return None
        
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        self.recorded_file = temp_file.name
        temp_file.close()
        
        # Save recording to WAV file
        with wave.open(self.recorded_file, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(self.frames))
        
        print(f"\nRecording saved: {len(self.frames)} frames, {os.path.getsize(self.recorded_file)} bytes")
        return self.recorded_file
    
    def cleanup(self):
        """Clean up resources"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        self.audio.terminate()
        
        if self.recorded_file and os.path.exists(self.recorded_file):
            os.unlink(self.recorded_file)

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
        print(f"Transcribing audio file: {audio_file_path}")
        
        with open(audio_file_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model=WHISPER_MODEL,  # Using model from screenshot
                file=audio_file
            )
        
        if not transcription.text:
            return ""
        
        print(f"You said: {transcription.text}")
        return transcription.text
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return ""

def get_ai_response(user_input):
    """Get AI response using the model from screenshot with conversation history"""
    global CONVERSATION_HISTORY
    print("Generating response...")
    
    try:
        # Prepare messages with conversation history
        messages = [
            {
                "role": "system", 
                "content": "You are a helpful, friendly voice assistant. Keep responses natural and concise."
            }
        ]
        
        # Add conversation history (last 10 exchanges for context)
        for entry in CONVERSATION_HISTORY[-10:]:
            messages.append(entry)
            
        # Add current user input
        messages.append({"role": "user", "content": user_input})
        
        # Use the model from the screenshot
        response = client.chat.completions.create(
            model=GPT_MODEL,
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
        print("Trying fallback model...")
        
        try:
            # Fallback to gpt-4o if realtime preview isn't available
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
        except Exception as fallback_error:
            print(f"Fallback also failed: {fallback_error}")
            return "I'm sorry, I couldn't process your request."

def speak_response(text, voice=DEFAULT_VOICE):
    """Convert text to speech using OpenAI TTS"""
    print("Generating speech...")
    
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

def main():
    """Main function to run the simple voice conversation"""
    global CONVERSATION_HISTORY
    
    print("=== Enhanced Voice Conversation System ===")
    print("Press ENTER to start recording")
    print("Press ENTER again to stop recording")
    print("Commands:")
    print("  'exit' - Quit the program")
    print("  'clear' - Clear conversation history")
    print("  'model' - Show current model settings")
    
    # Select microphone
    mic_index = list_microphones()
    if mic_index is None:
        print("No microphone available. Exiting.")
        return
    
    recorder = Recorder()
    
    try:
        while True:
            command = input("\nPress ENTER to start recording (or type a command): ").strip().lower()
            
            if command == 'exit':
                break
            elif command == 'clear':
                CONVERSATION_HISTORY = []
                print("Conversation history cleared.")
                continue
            elif command == 'model':
                print(f"Current models:")
                print(f"  Chat: {GPT_MODEL} (temperature: {GPT_TEMPERATURE})")
                print(f"  Transcription: {WHISPER_MODEL}")
                print(f"  TTS: tts-1 (voice: {DEFAULT_VOICE})")
                print(f"  Conversation history: {len(CONVERSATION_HISTORY)//2} exchanges")
                continue
            
            # Start recording in a separate thread
            import threading
            recording_thread = threading.Thread(target=recorder.start_recording, args=(mic_index,))
            recording_thread.daemon = True
            recording_thread.start()
            
            # Wait for ENTER to stop recording
            input("")
            recorder.is_recording = False
            recording_thread.join()
            
            # Get the recording file
            audio_file = recorder.stop_recording()
            if not audio_file:
                print("No audio recorded or file not found.")
                continue
            
            # Transcribe
            transcript = transcribe_audio(audio_file)
            if not transcript:
                print("Couldn't transcribe audio. Please try again.")
                continue
            
            # Get AI response
            response = get_ai_response(transcript)
            
            # Speak the response
            speak_response(response)
            
            # Clean up the audio file
            if os.path.exists(audio_file):
                os.unlink(audio_file)
    
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        recorder.cleanup()

if __name__ == "__main__":
    main() 