import argparse
import whisper  # https://github.com/lablab-ai/Whisper-transcription_and_diarization-speaker-identification-, https://lablab.ai/t/whisper-transcription-and-speaker-identification
from pathlib import Path
import subprocess
import os
import datetime
import soundfile as sf
from simple_diarizer.diarizer import (
    Diarizer,
)  # https://github.com/cvqluu/simple_diarizer
from pydub import AudioSegment
import warnings

# Argparse stuff
parser = argparse.ArgumentParser(
    prog="Transcribe Video",
    description="This program takes the path to a video file and transcribes it using Whisper from OpenAI",
)

parser.add_argument(
    "-n",
    "--number_speakers",
    type=int,
    help="The number of speakers in the video",
    default=2,
)
parser.add_argument("file_path", nargs="+")
args = parser.parse_args()

# Diarization and transcription model initialisation
model = whisper.load_model("small.en")  # tiny.en, base.en, small.en, medium.en
diar = Diarizer(
    embed_model="xvec",  # 'xvec' and 'ecapa' supported
    cluster_method="sc",  # 'ahc' and 'sc' supported
)

for file_path_item in args.file_path:
    # File paths
    file_name = Path(file_path_item).stem
    file_parent = Path(file_path_item).parents[0]
    # mp3_audio_file_path = Path(file_parent).joinpath(f"{file_name}.mp3")
    wav_audio_file_path = Path(file_parent).joinpath(f"{file_name}.wav")
    wav_segment_file_path = Path(file_parent).joinpath(f"{file_name}_segment.wav")

    # Generate an mp3 and wav
    # command = f"ffmpeg -i '{file_path_item}' '{mp3_audio_file_path}'"
    # subprocess.call(command, shell=True)
    command = f"ffmpeg -y -i '{file_path_item}' -acodec pcm_s16le -ar 16000 -ac 1 '{wav_audio_file_path}'"
    subprocess.call(command, shell=True)

    # Diarization
    diarization = diar.diarize(
        str(wav_audio_file_path), num_speakers=args.number_speakers
    )

    # Clean diarization results
    master_dictionary = []
    first_speaker = -1
    current_speaker = -1
    current_speaker_start = -1
    for segment in diarization:
        segment.pop("start_sample")
        segment.pop("end_sample")
        if first_speaker == -1:
            first_speaker = segment["label"]
            current_speaker = segment["label"]
            current_speaker_start = segment["start"]
        if segment["label"] != current_speaker:
            dict_to_add = {
                "start": current_speaker_start,
                "end": segment["start"],
                "label": current_speaker,
            }
            master_dictionary.append(dict_to_add)
            current_speaker = segment["label"]
            current_speaker_start = segment["start"]

    # Re-lable the speakers to be more friendly
    speaker_labels = set()
    [speaker_labels.add(dz_segment["label"]) for dz_segment in master_dictionary]
    speaker_labels = list(speaker_labels)
    speaker_labels = [str(label) for label in speaker_labels]

    new_speaker_labels = []
    for i in range(1, (len(speaker_labels) + 1)):
        new_speaker_labels.append(f"Person {i}")
    mapping = dict(zip(speaker_labels, new_speaker_labels))
    for segment in master_dictionary:
        old_label = str(segment["label"])
        segment["label"] = mapping[old_label]
    
    # Main itteration over speakers
    print("Transcribing...")
    for speaker in master_dictionary:
        speaker_start_ms = speaker["start"]*1000
        speaker_end_ms = speaker["end"]*1000
        wav_segment = AudioSegment.from_wav(wav_audio_file_path)[speaker_start_ms:speaker_end_ms].export(wav_segment_file_path, format="wav") 
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            transcribed_segment = model.transcribe(str(wav_segment_file_path))["segments"]
        os.remove(wav_segment_file_path)
        all_transcribed_lines = []
        for line in transcribed_segment:
            all_transcribed_lines.append(line["text"])
        speaker_transcription = "".join(all_transcribed_lines)
        speaker["transcription"] = speaker_transcription.lstrip()

    # Build a new text file to construct into the output
    transcribed_text_list = []
    for i, speaker in enumerate(master_dictionary):
        # Get the data to add to the speaker and timestamp line
        new_speaker = speaker["label"]
        new_speaker_start_time = speaker["start"]
        new_speaker_start_time = f"{float(new_speaker_start_time):.0f}"  # Format seconds to whole numbers
        new_speaker_start_time = str(
            datetime.timedelta(seconds=float(new_speaker_start_time))
        )  # Format as H:M:S

        # Add the speaker and timestamp line
        if 1 == 0:
            transcribed_text_list.append(
                f"{new_speaker} [{new_speaker_start_time}]\n"
            )
        else:
            transcribed_text_list.append(
                f"\n\n{new_speaker} [{new_speaker_start_time}]\n"
            )

        # Add the transcribed content
        content = speaker["transcription"]
        transcribed_text_list.append(f"{content}")

    # Post process the list of segments and speaker lines into a single string and clean
    transcribed_text = "".join(transcribed_text_list)
    transcribed_text = transcribed_text.replace("\n ", "\n").lstrip()

    # Create text file
    text_file_path = Path(file_parent).joinpath(f"{file_name}.txt")
    file = open(text_file_path, "w")
    file.write(transcribed_text)

    # Delete the mp3 and wav
    # os.remove(mp3_audio_file_path)
    os.remove(wav_audio_file_path)