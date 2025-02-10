import openai
import pyaudio
import wave
import keyboard
import time

def record_audio(output_filename="test.wav"):
    """
    Record audio from microphone until user presses space, then save as WAV.
    """
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 1024

    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=FORMAT, 
        channels=CHANNELS, 
        rate=RATE, 
        input=True, 
        frames_per_buffer=CHUNK
    )

    frames = []
    print("Press Space to start recording.")
    keyboard.wait("space")
    print("Recording... Press Space to stop recording.")
    time.sleep(0.2)  # slight delay so we don't cut off audio

    while True:
        try:
            data = stream.read(CHUNK)
            frames.append(data)
        except KeyboardInterrupt:
            # In case the user hits Ctrl+C
            break
        if keyboard.is_pressed("space"):
            print("Recording stopped.")
            time.sleep(0.2)
            break

    stream.stop_stream()
    stream.close()
    audio.terminate()

    with wave.open(output_filename, 'wb') as wave_file:
        wave_file.setnchannels(CHANNELS)
        wave_file.setsampwidth(audio.get_sample_size(FORMAT))
        wave_file.setframerate(RATE)
        wave_file.writeframes(b''.join(frames))

def speech_to_text(
    audio_filename="test.wav", 
    api_key="YOUR_OPENAI_API_KEY"
) -> str:
    """
    Transcribe an audio file (WAV) using OpenAI's Whisper model.
    
    :param audio_filename: Path to the WAV file.
    :param api_key: Your OpenAI API key.
    :return: The transcription text.
    """
    openai.api_key = api_key
    with open(audio_filename, "rb") as audio_file:
        transcription = openai.Audio.translations.create(
            model="whisper-1",
            file=audio_file,
        )
    return transcription["text"]

if __name__ == "__main__":
    # Example usage:
    record_audio("test.wav")
    text = speech_to_text("test.wav")
    print("Transcribed text:", text)
