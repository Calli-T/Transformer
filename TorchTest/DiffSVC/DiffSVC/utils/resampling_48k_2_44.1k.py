import os
from pydub import AudioSegment

def resample_audio_files(base_dir, target_sr=44100):
    if not os.path.exists(base_dir):
        print(f"Base directory '{base_dir}' not found.")
        return

    for speaker_name in os.listdir(base_dir):
        speaker_dir = os.path.join(base_dir, speaker_name)

        if os.path.isdir(speaker_dir):
            print(f"Processing speaker: {speaker_name}")
            for file_name in os.listdir(speaker_dir):
                file_path = os.path.join(speaker_dir, file_name)

                if os.path.isfile(file_path) and file_name.lower().endswith(('.wav', '.mp3', '.flac')):
                    try:
                        audio = AudioSegment.from_file(file_path)

                        if audio.frame_rate == 48000:
                            resampled_audio = audio.set_frame_rate(target_sr)
                            resampled_audio.export(file_path, format="wav")
                            print(f"Resampled and replaced: {file_name}")
                        else:
                            print(f"Skipped (not 48kHz): {file_name}")

                    except Exception as e:
                        print(f"Error processing file {file_name}: {e}")

if __name__ == "__main__":
    base_directory = input("Enter the base directory: ").strip()
    resample_audio_files(base_directory)