"""
Simple Streaming Voice Conversation with OpenAI
Press ENTER to start recording, press ENTER again to stop
"""

import os
import tempfile
import time
import wave
import pyaudio
import numpy as np
import subprocess
import threading
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

# Models from screenshot
GPT_MODEL = "gpt-4o"
WHISPER_MODEL = "whisper-1"
GPT_TEMPERATURE = 0.8

# Conversation history
CONVERSATION_HISTORY = []

class Recorder:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.frames = []
        self.is_recording = False
        self.recorded_file = None
    
    def record(self, device_index=None):
        """Record audio until user presses Enter"""
        self.frames = []
        self.is_recording = True
        
        # Setup the stream
        self.stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=CHUNK
        )
        
        print("Recording started... Press ENTER to stop.")
        
        # Start recording thread
        def recording_thread():
            while self.is_recording:
                try:
                    data = self.stream.read(CHUNK, exception_on_overflow=False)
                    self.frames.append(data)
                    
                    # Show volume meter
                    audio_data = np.frombuffer(data, dtype=np.int16)
                    volume_norm = np.linalg.norm(audio_data) / 100
                    meter = "=" * int(min(volume_norm, 50)) + "_" * (50 - int(min(volume_norm, 50)))
                    print(f"\r🎤 Recording [{meter}] {volume_norm:.1f}", end="", flush=True)
                    
                    time.sleep(0.01)  # Prevent CPU overuse
                except Exception as e:
                    print(f"\nError during recording: {e}")
                    break
        
        # Start the recording thread
        rec_thread = threading.Thread(target=recording_thread)
        rec_thread.daemon = True
        rec_thread.start()
        
        # Wait for user to press Enter to stop recording
        input("")
        self.stop()
        
        return self.save_recording()
    
    def stop(self):
        """Stop recording"""
        self.is_recording = False
        
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
            except Exception as e:
                print(f"\nError stopping stream: {e}")
    
    def save_recording(self):
        """Save the recording to a WAV file"""
        if not self.frames or len(self.frames) < 3:
            print("\nRecording too short")
            return None
        
        try:
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
        except Exception as e:
            print(f"\nError saving recording: {e}")
            return None
    
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
        
        if self.recorded_file and os.path.exists(self.recorded_file):
            try:
                os.unlink(self.recorded_file)
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

def transcribe_audio(audio_file):
    """Transcribe audio file using OpenAI's Whisper model"""
    try:
        print("Transcribing audio...")
        
        # Open the audio file and transcribe it
        with open(audio_file, "rb") as f:
            transcription = client.audio.transcriptions.create(
                model=WHISPER_MODEL,
                file=f
            )
        
        if transcription.text:
            result = transcription.text.strip()
            print(f"You said: {result}")
            return result
        else:
            print("Could not transcribe audio. No speech detected.")
            return None
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None

def generate_response(user_input):
    """Generate a response using streaming"""
    global CONVERSATION_HISTORY
    
    if not user_input:
        return "I couldn't understand that. Could you try again?"
    
    print("Generating response...")
    
    try:
        # Prepare the conversation history
        messages = [
            {"role": "system", "content": "You are a helpful, friendly voice assistant. Keep responses natural and concise."}
        ]
        
        # Add conversation history
        for entry in CONVERSATION_HISTORY[-10:]:
            messages.append(entry)
        
        # Add the new user message
        messages.append({"role": "user", "content": user_input})
        
        # Stream the response with a standard model
        print("Assistant: ", end="", flush=True)
        response_content = ""
        
        # Create a streaming response with gpt-4o (standard model)
        stream = client.chat.completions.create(
            model=GPT_MODEL,
            messages=messages,
            temperature=GPT_TEMPERATURE,
            stream=True
        )
        
        # Process the streaming response
        for chunk in stream:
            if chunk.choices[0].delta.content:
                content_piece = chunk.choices[0].delta.content
                response_content += content_piece
                print(content_piece, end="", flush=True)
        
        print()  # Add a newline at the end
        
        # Add to conversation history
        CONVERSATION_HISTORY.append({"role": "user", "content": user_input})
        CONVERSATION_HISTORY.append({"role": "assistant", "content": response_content})
        
        return response_content
            
    except Exception as e:
        print(f"Error generating response: {e}")
        return "I'm sorry, I couldn't process your request. Please try again."

def speak_response(text):
    """Convert text to speech using OpenAI's Text-to-Speech API"""
    if not text:
        return
    
    print("Converting to speech...")
    
    try:
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        temp_filename = temp_file.name
        temp_file.close()
        
        # Generate speech
        speech_response = client.audio.speech.create(
            model="tts-1",
            voice=DEFAULT_VOICE,
            input=text,
            speed=1.1
        )
        
        # Save to file
        with open(temp_filename, "wb") as f:
            f.write(speech_response.content)
        
        # Play using system command
        if os.name == 'posix':  # macOS or Linux
            subprocess.run(['afplay' if 'darwin' in os.uname().sysname.lower() else 'aplay', temp_filename], 
                         check=True)
        elif os.name == 'nt':  # Windows
            os.startfile(temp_filename)
        
        # Clean up
        time.sleep(0.5)
        if os.path.exists(temp_filename):
            os.unlink(temp_filename)
            
    except Exception as e:
        print(f"Error speaking response: {e}")

def main():
    """Main function to run the voice conversation system"""
    global CONVERSATION_HISTORY
    
    print("=== Simple Streaming Voice Conversation ===")
    print("Press ENTER to start recording")
    print("Press ENTER again to stop recording")
    print("Type 'exit' to quit, 'clear' to clear conversation history")
    
    # Select microphone
    mic_index = list_microphones()
    if mic_index is None:
        print("No microphone available. Exiting.")
        return
    
    recorder = Recorder()
    
    try:
        while True:
            command = input("\nPress ENTER to start recording (or type a command): ").strip().lower()
            
            if command == 'exit' or command == 'quit':
                break
            elif command == 'clear':
                CONVERSATION_HISTORY = []
                print("Conversation history cleared.")
                continue
            elif command == 'history':
                print("\nConversation History:")
                for i, entry in enumerate(CONVERSATION_HISTORY):
                    role = entry["role"]
                    content = entry["content"]
                    print(f"{i+1}. {role.capitalize()}: {content[:50]}{'...' if len(content) > 50 else ''}")
                continue
            
            # Record audio
            audio_file = recorder.record(mic_index)
            
            if not audio_file:
                print("No valid recording found. Please try again.")
                continue
            
            # Transcribe audio
            transcript = transcribe_audio(audio_file)
            
            if not transcript:
                if os.path.exists(audio_file):
                    os.unlink(audio_file)
                continue
            
            # Generate response
            response = generate_response(transcript)
            
            # Speak the response
            speak_response(response)
            
            # Clean up audio file
            if os.path.exists(audio_file):
                os.unlink(audio_file)
    
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error in main loop: {e}")
    finally:
        recorder.cleanup()

if __name__ == "__main__":
    main() 