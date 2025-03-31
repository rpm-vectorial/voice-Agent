import pyaudio
import numpy as np
import time

# Audio recording settings
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 24000
TEST_DURATION = 10  # seconds

def test_microphone():
    """Test microphone and display input levels"""
    print("Starting microphone test...")
    
    # Initialize PyAudio
    p = pyaudio.PyAudio()
    
    # List available microphones
    print("\nAvailable input devices:")
    for i in range(p.get_device_count()):
        dev_info = p.get_device_info_by_index(i)
        if dev_info.get('maxInputChannels') > 0:
            print(f"Device {i}: {dev_info.get('name')}")
    
    # Ask user to select a microphone
    device_index = int(input("\nEnter device index to test (or press Enter for default): ") or "-1")
    
    # Open stream
    try:
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=device_index if device_index >= 0 else None,
            frames_per_buffer=CHUNK
        )
        
        print(f"\nTesting microphone {'(default)' if device_index < 0 else f'(device {device_index})'}")
        print("Speak into your microphone to see the volume levels.")
        print(f"Test will run for {TEST_DURATION} seconds.")
        print("Press Ctrl+C to stop the test early.")
        
        # Record and show volume levels
        start_time = time.time()
        try:
            while time.time() - start_time < TEST_DURATION:
                data = stream.read(CHUNK, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.int16)
                volume_norm = np.linalg.norm(audio_data) / 100
                
                # Print volume meter
                meter = "=" * int(min(volume_norm, 50))
                print(f"\rVolume: [{'=' * int(min(volume_norm, 50))}{' ' * (50 - int(min(volume_norm, 50)))}] {volume_norm:.1f}", end="", flush=True)
                
                # Indicate if speech would be detected with current settings
                if volume_norm > 200:  # Same as SILENCE_THRESHOLD in voice_conversation.py
                    print(" (SPEECH DETECTED)", end="", flush=True)
                else:
                    print("                  ", end="", flush=True)
                    
                time.sleep(0.05)
        except KeyboardInterrupt:
            print("\nTest stopped by user.")
    
    except Exception as e:
        print(f"\nError testing microphone: {e}")
    
    finally:
        # Clean up
        if 'stream' in locals():
            stream.stop_stream()
            stream.close()
        p.terminate()
        print("\nMicrophone test complete.")

if __name__ == "__main__":
    test_microphone() 