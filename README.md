# Voice Agent

A powerful voice conversation system that enables natural voice interactions using OpenAI's APIs. This system serves as the interfacing agent for an agentic framework, allowing users to interact with AI through speech.

## Features

- **Real-time Speech Detection**: Accurately detects when a user starts and stops speaking
- **Natural Conversation Flow**: Supports interruptions and maintains context
- **High-quality Voice Synthesis**: Uses OpenAI's Text-to-Speech for natural-sounding responses
- **Multi-modal Termination**: Multiple ways to stop the conversation (voice commands, keyboard, file-based)
- **Graceful Resource Management**: Proper cleanup of resources when terminating
- **Integrated Utilities**: Built-in tools for control and management of the voice agent

## Architecture

The Voice Agent is structured as a pipeline of specialized components:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Microphone     │───>│  Speech         │───>│  Language       │───>│  Text-to-Speech │
│  (Audio Input)  │    │  Recognition    │    │  Model          │    │  Synthesis      │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
        │                      │                      │                       │
        │                      │                      │                       │
        v                      v                      v                       v
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           Voice Activity Detection & Control                         │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### Components

- **Audio Input Service**: Captures audio from the microphone and detects speech
- **Voice Activity Detection**: Analyzes audio to determine when speaking starts and stops
- **Speech-to-Text Service**: Transcribes speech to text using OpenAI's Whisper model
- **Language Model Service**: Generates responses using OpenAI's GPT-4o
- **Text-to-Speech Service**: Converts responses to speech using OpenAI's TTS
- **Voice Agent Utilities**: Provides integrated tools for controlling the agent

## Requirements

- Python 3.9+
- OpenAI API key
- Microphone and speakers
- Operating system: Windows, macOS, or Linux

## Installation

1. Clone the repository
   ```bash
   git clone https://github.com/yourusername/Voice-Agent.git
   cd Voice-Agent
   ```

2. Create and activate a virtual environment
   ```bash
   # On Windows
   python -m venv voice-agent
   voice-agent\Scripts\activate
   
   # On macOS/Linux
   python -m venv voice-agent
   source voice-agent/bin/activate
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root with your OpenAI API key
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

The Voice Agent can be run in several ways:

### Standard Mode

Run the voice agent with manual termination:

```bash
python openai_pipecat.py
```

You can stop the conversation by:
- Saying "stop", "exit", "quit", or similar termination phrases
- Pressing Ctrl+C
- Running `python openai_pipecat.py stop`

### Timed Mode

Run the voice agent with automatic termination after a specified time:

```bash
# Run for 60 seconds (default)
python timed_conversation.py

# Run for a custom duration (e.g., 120 seconds)
python timed_conversation.py --time 120
```

### Emergency Termination

If the voice agent doesn't respond, you can use built-in kill functionality:

```bash
# Gracefully stop the voice agent
python openai_pipecat.py stop

# Force stop the voice agent 
python openai_pipecat.py stop --force

# Kill all Python processes
python openai_pipecat.py kill
```

## Configuration

The Voice Agent can be configured by modifying the following classes in `openai_pipecat.py`:

### Audio Configuration

```python
class AudioConfig:
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 24000
    SILENCE_THRESHOLD = 1000  # Adjust for speech detection sensitivity
    SILENCE_DURATION = 0.8    # How long silence must persist to end utterance
    CONVERSATION_TIMEOUT = 8.0  # Seconds of silence to end conversation
    INTERRUPTION_THRESHOLD = 1200  # Threshold for interruption detection
```

### AI Model Configuration

```python
class AIConfig:
    STT_MODEL = "whisper-1"    # Speech-to-text model
    LLM_MODEL = "gpt-4o"       # Language model
    TTS_MODEL = "tts-1"        # Text-to-speech model
    TTS_VOICE = "nova"         # Voice for speech synthesis
    TEMPERATURE = 0.7          # Response creativity
    SYSTEM_PROMPT = "..."      # System prompt for the language model
```

## Troubleshooting

### Voice Not Detected

If your voice is not being detected:
- Try increasing the microphone volume in your system settings
- Decrease the `SILENCE_THRESHOLD` in the `AudioConfig` class
- Make sure your microphone is working and properly connected

### AI Responses Not Interrupted

If you can't interrupt the AI's responses:
- Decrease the `INTERRUPTION_THRESHOLD` in the `AudioConfig` class
- Speak louder when interrupting

### Process Not Terminating

If the voice agent doesn't terminate properly:
- Use `python openai_pipecat.py stop --force` to force termination
- As a last resort, use `python openai_pipecat.py kill` to kill all Python processes

## Development

### Code Structure

- `openai_pipecat.py`: Main voice agent implementation with integrated control functionality
- `voice_agent_utils.py`: Utility functions for controlling the voice agent
- `timed_conversation.py`: Run the voice agent with a time limit

### Adding Features

To extend the Voice Agent, follow these guidelines:

1. Keep each component focused on a single responsibility
2. Maintain error handling and resource cleanup
3. Update documentation for new features
4. Add appropriate type hints for new functions and methods

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -am 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgements

- OpenAI for providing the API services
- PyAudio for audio processing capabilities 