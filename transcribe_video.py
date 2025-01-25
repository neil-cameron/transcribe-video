import argparse
import os
from faster_whisper import WhisperModel
from pathlib import Path
import datetime
from pyannote.audio import Pipeline  # For speaker diarization
from pydub import AudioSegment
from io import BytesIO
import warnings
import summarise
import time
import config
import ffmpeg  # For in-memory audio extraction
import torch

# Start timing
start_time = time.time()

# Argparse setup
parser = argparse.ArgumentParser(
    prog="Transcribe Video",
    description="This program takes the path to a video file and transcribes it using Faster-Whisper with automatic speaker diarization using Pyannote.audio",
)

parser.add_argument(
    "file_path",
    nargs="+",
    help="A full or relative path to a media file, several media files, or a directory of media files to transcribe",
)
args = parser.parse_args()

# Parse arguments
argparse_list_of_paths = []
if args.file_path:
    [
        argparse_list_of_paths.append(individual_path)
        for individual_path in args.file_path
    ]

full_file_path_list = []  # This is sent to the main transcribing function
for individual_path in argparse_list_of_paths:
    if os.path.isdir(individual_path):  # Directory
        for dir_path, dir_names, file_names in os.walk(individual_path):
            for file_name in file_names:
                if not file_name.startswith("."):
                    file_path_found = os.path.join(dir_path, file_name)
                    full_file_path_list.append(file_path_found)
    else:  # File
        full_file_path_list.append(individual_path)


# Utility function to extract audio from video in memory using ffmpeg-python
def extract_audio_from_video(video_path, target_sample_rate=16000):
    """
    Extract audio from a video file in memory and return it as a pydub AudioSegment.
    """
    out, _ = (
        ffmpeg.input(video_path)
        .output("pipe:1", format="wav", ac=1, ar=target_sample_rate)
        .run(capture_stdout=True, capture_stderr=True)
    )
    return AudioSegment.from_file(BytesIO(out), format="wav")


# Diarization and transcription model initialization
model = WhisperModel("small", device="cpu")  # Faster-Whisper does not support mps

# Determine device for PyTorch (mps or cpu)
device = (
    torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
)

# Initialize Pyannote pipeline and set device
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization", use_auth_token=config.hf_authorization
)
pipeline.to(device)  # Set the device for the pipeline

for file_counter, file_path_item in enumerate(full_file_path_list):
    # File paths
    file_name = Path(file_path_item).stem
    file_parent = Path(file_path_item).parents[0]

    # Check if the input is a video or audio file
    if file_path_item.lower().endswith((".mp4", ".mkv", ".avi", ".mov")):
        # Extract audio from video
        print(f"Extracting audio from video: {file_path_item}...")
        audio = extract_audio_from_video(file_path_item)
    else:
        # Load audio directly
        audio = (
            AudioSegment.from_file(file_path_item).set_frame_rate(16000).set_channels(1)
        )

    # Convert the in-memory `pydub.AudioSegment` into a BytesIO file-like object
    audio_buffer = BytesIO()
    audio.export(audio_buffer, format="wav")
    audio_buffer.seek(0)

    # Speaker diarization using pyannote.audio
    print(f"Diarizing file {file_counter+1} of {len(full_file_path_list)}...")
    diarization = pipeline({"uri": file_name, "audio": audio_buffer})

    # Parse diarization results into a clean format
    # Create a mapping for speaker labels
    speaker_mapping = {}
    speaker_count = 1

    master_dictionary = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if speaker not in speaker_mapping:
            speaker_mapping[speaker] = f"Speaker {speaker_count}"
            speaker_count += 1
        master_dictionary.append(
            {"start": turn.start, "end": turn.end, "label": speaker_mapping[speaker]}
        )

    # Group contiguous segments by speaker
    grouped_segments = []
    current_speaker = None
    current_group = {"speaker": None, "start": None, "end": None}

    for segment in master_dictionary:
        if segment["label"] != current_speaker:
            if current_speaker is not None:
                grouped_segments.append(current_group)
            current_speaker = segment["label"]
            current_group = {
                "speaker": current_speaker,
                "start": segment["start"],
                "end": segment["end"],
            }
        else:
            current_group["end"] = segment["end"]  # Extend the end time of the group
    grouped_segments.append(current_group)  # Add the last group

    # Transcribe grouped segments
    print(f"Transcribing file {file_counter+1} of {len(full_file_path_list)}...")
    for group in grouped_segments:
        speaker = group["speaker"]
        speaker_start_ms = int(group["start"] * 1000)
        speaker_end_ms = int(group["end"] * 1000)

        # Extract the audio segment in memory
        wav_segment = audio[speaker_start_ms:speaker_end_ms]

        # Export the audio segment to a BytesIO object
        audio_buffer = BytesIO()
        wav_segment.export(audio_buffer, format="wav")
        audio_buffer.seek(0)  # Move to the start of the buffer

        # Transcribe the in-memory audio segment
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            segments, info = model.transcribe(audio_buffer, beam_size=5, language="en")

        # Combine transcribed text
        all_transcribed_lines = [segment.text.strip() for segment in segments]
        transcription = " ".join(all_transcribed_lines).strip()

        # Add transcription to the group's data
        group["transcription"] = transcription

    # Construct output text
    transcribed_text_list = []
    for group in grouped_segments:
        new_speaker = group["speaker"]
        new_speaker_start_time = str(
            datetime.timedelta(seconds=int(group["start"]))
        )  # Format as H:M:S

        # Add the speaker and timestamp line
        transcribed_text_list.append(f"\n\n{new_speaker} [{new_speaker_start_time}]\n")
        transcribed_text_list.append(group["transcription"])

    # Post-process the list into a single string
    transcribed_text = "".join(transcribed_text_list).strip()

    # Summarize the transcribed text
    summary_length = 300
    summary = summarise.summarise(transcribed_text, summary_length)

    # Extract meeting actions
    actions = summarise.find_actions(transcribed_text)
    summarised_transcribed_text = "\n\n\n".join([summary, actions, transcribed_text])

    # Save to text file
    text_file_path = Path(file_parent).joinpath(f"{file_name}.txt")
    with open(text_file_path, "w") as file:
        file.write(summarised_transcribed_text)

    # End timing
    elapsed_time = time.time() - start_time
    elapsed_minutes = int(elapsed_time / 60)
    elapsed_seconds = int(elapsed_time % 60)
    print(f"Elapsed time: {elapsed_minutes:.0f}:{elapsed_seconds:.2f}")
