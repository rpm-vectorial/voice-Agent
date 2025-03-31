import pyaudio
import audioop
import time
import wave
import tempfile
import os
from openai import OpenAI

# Audio settings
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 24000
SILENCE_THRESHOLD = 100  # Very low threshold
SILENCE_DURATION = 1.0  # seconds of silence to end speaking

def record_speech():
    """Record speech from microphone with basic VAD"""
    print("=== Simple Speech Detection Test ===")
    print("This test will record when you speak and stop when you pause.")
    print("Start speaking when ready...")
    
    # Initialize PyAudio
    p = pyaudio.PyAudio()
    
    # Open audio stream
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )
    
    # Track frames
    frames = []
    speaking = False
    silence_start = None
    
    try:
        print("\nListening...")
        
        # Wait for initial speech
        while not speaking:
            chunk = stream.read(CHUNK, exception_on_overflow=False)
            rms = audioop.rms(chunk, 2)
            
            # Print current volume
            print(f"\rVolume: {rms:.1f} | Threshold: {SILENCE_THRESHOLD}", end="", flush=True)
            
            # Check if volume exceeds threshold
            if rms > SILENCE_THRESHOLD:
                speaking = True
                frames.append(chunk)
                print("\nSpeech detected! Recording...")
        
        # Record until silence
        while True:
            chunk = stream.read(CHUNK, exception_on_overflow=False)
            rms = audioop.rms(chunk, 2)
            
            # Visual feedback during recording
            bars = min(40, int(rms / 30))
            print(f"\r[{'|' * bars}{' ' * (40-bars)}] {rms:.1f}", end="", flush=True)
            
            # Add frame to the recording
            frames.append(chunk)
            
            # Check for silence to end recording
            if rms <= SILENCE_THRESHOLD:
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start > SILENCE_DURATION:
                    print("\nSilence detected, stopping recording.")
                    break
            else:
                silence_start = None
        
    except KeyboardInterrupt:
        print("\nRecording stopped by user.")
    finally:
        # Clean up
        stream.stop_stream()
        stream.close()
        p.terminate()
    
    return frames

def transcribe_audio(frames):
    """Save frames to a WAV file and transcribe"""
    # Create temporary WAV file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        filename = temp_file.name
    
    # Save audio to WAV file
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 2 bytes for FORMAT=paInt16
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
    
    try:
        # Transcribe using OpenAI
        client = OpenAI()
        with open(filename, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file, 
                language="en"
            )
        
        text = transcription.text.strip()
        print(f"Transcription: {text}")
        return text
    except Exception as e:
        print(f"Error transcribing: {e}")
        return ""
    finally:
        # Clean up
        if os.path.exists(filename):
            os.unlink(filename)

def main():
    """Main function"""
    # Record speech
    frames = record_speech()
    
    if frames:
        print("Processing speech...")
        transcribe_audio(frames)
    else:
        print("No speech detected.")

if __name__ == "__main__":
    main() 