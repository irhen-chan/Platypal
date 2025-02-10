import openai
from pathlib import Path

def text_to_speech(
    text: str, 
    filename: str = "speech.mp3", 
    voice: str = "alloy", 
    api_key: str = "YOUR_OPENAI_API_KEY"
) -> str:
    """
    Convert text to speech using OpenAI's TTS model and save as MP3.

    :param text: The text to synthesize.
    :param filename: The name of the output MP3 file.
    :param voice: The voice model to use for TTS (e.g., "alloy").
    :param api_key: Your OpenAI API key.
    :return: The path to the MP3 file.
    """
    openai.api_key = api_key

    response = openai.Audio.speech.create(
        model="tts-1",      # e.g., "tts-1" or any other TTS model
        voice=voice,
        input=text,
    )
    # The response is streamed back in chunks. Write them out to file:
    with open(filename, "wb") as f:
        for chunk in response:
            f.write(chunk)

    return filename

if __name__ == "__main__":
    # Quick test:
    mp3_path = text_to_speech("Hello world! This is a test of text-to-speech.")
    print(f"TTS output saved to {mp3_path}")
