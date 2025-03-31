"""
Real-time OpenAI Voice Conversation
Based on Microsoft example but using standard OpenAI API
"""

import os
import base64
import asyncio
import tempfile
from openai import OpenAI
from dotenv import load_dotenv
import subprocess

# Load environment variables
load_dotenv()

# Ensure API key is available
if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError("OPENAI_API_KEY environment variable is not set")

# Initialize OpenAI client
client = OpenAI()

async def main() -> None:
    """
    Simple voice conversation using standard OpenAI API.
    When prompted for user input, type a message and hit enter to send it to the model.
    Enter "q" to quit the conversation.
    """
    print("=== Real-time Voice Conversation with OpenAI ===")
    print("Type a message and hit Enter to talk to the AI")
    print("Enter 'q' to quit the conversation")
    print()
    
    # Keep track of conversation history
    messages = [
        {"role": "system", "content": "You are a helpful, friendly assistant. Keep responses natural and concise."}
    ]
    
    while True:
        # Get user input
        user_input = input("You: ")
        if user_input.lower() == "q":
            print("Exiting...")
            break
        
        # Add to messages
        messages.append({"role": "user", "content": user_input})
        
        # Generate streaming text response
        print("Assistant: ", end="", flush=True)
        stream = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.8,
            stream=True
        )
        
        response_content = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                content_piece = chunk.choices[0].delta.content
                response_content += content_piece
                print(content_piece, end="", flush=True)
        
        print()  # Add a newline at the end
        
        # Add the assistant's response to the messages
        messages.append({"role": "assistant", "content": response_content})
        
        # Generate speech from the response
        try:
            # Create a temporary file
            temp_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
            temp_filename = temp_file.name
            temp_file.close()
            
            # Generate speech
            speech_response = client.audio.speech.create(
                model="tts-1",
                voice="nova",
                input=response_content,
                speed=1.1
            )
            
            # Save to file
            with open(temp_filename, "wb") as f:
                f.write(speech_response.content)
            
            # Play using system command (Mac or Linux)
            if os.name == 'posix':
                subprocess.run(['afplay' if 'darwin' in os.uname().sysname.lower() else 'aplay', temp_filename], 
                              check=True)
            
            # Clean up
            os.unlink(temp_filename)
                
        except Exception as e:
            print(f"Error speaking response: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 