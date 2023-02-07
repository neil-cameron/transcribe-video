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
    mp3_audio_file_path = Path(file_parent).joinpath(f"{file_name}.mp3")
    wav_audio_file_path = Path(file_parent).joinpath(f"{file_name}.wav")

    # Generate an mp3 and wav
    command = f"ffmpeg -i '{file_path_item}' '{mp3_audio_file_path}'"
    subprocess.call(command, shell=True)
    command = f"ffmpeg -y -i '{file_path_item}' -acodec pcm_s16le -ar 16000 -ac 1 '{wav_audio_file_path}'"
    subprocess.call(command, shell=True)

    # Diarization
    diarization = diar.diarize(
        str(wav_audio_file_path), num_speakers=args.number_speakers
    )

    # Clean diarization results
    clean_dz_segment_list = []
    first_speaker = -1
    current_speaker = -1
    current_speaker_start = -1
    for dz_segment in diarization:
        dz_segment.pop("start_sample")
        dz_segment.pop("end_sample")
        if first_speaker == -1:
            first_speaker = dz_segment["label"]
            current_speaker = dz_segment["label"]
            current_speaker_start = dz_segment["start"]
        if dz_segment["label"] != current_speaker:
            dict_to_add = {
                "start": current_speaker_start,
                "end": dz_segment["start"],
                "label": current_speaker,
            }
            clean_dz_segment_list.append(dict_to_add)
            current_speaker = dz_segment["label"]
            current_speaker_start = dz_segment["start"]

    # Re-lable the speakers to be more friendly
    speaker_labels = set()
    [speaker_labels.add(dz_segment["label"]) for dz_segment in clean_dz_segment_list]
    speaker_labels = list(speaker_labels)
    speaker_labels = [str(label) for label in speaker_labels]

    new_speaker_labels = []
    for i in range(1, (len(speaker_labels) + 1)):
        new_speaker_labels.append(f"Person {i}")
    mapping = dict(zip(speaker_labels, new_speaker_labels))
    for dz_segment in clean_dz_segment_list:
        old_label = str(dz_segment["label"])
        dz_segment["label"] = mapping[old_label]

    # Transcribe
    transcribed_text_segment_dict_list = model.transcribe(str(mp3_audio_file_path))[
        "segments"
    ]

    # Build a new text file including time stamps line by line
    transcribed_text_lines_list = []
    current_dz_segment_list_index = -1
    for i, tr_segment in enumerate(transcribed_text_segment_dict_list):
        # First line of the file inc first speaker
        if i == 0:
            first_speaker = clean_dz_segment_list[0]["label"]
            transcribed_text_lines_list.append(f"{first_speaker} [00:00:00]\n")
            current_dz_segment_list_index += 1

        # First segment onwards
        # Segment falls within the current speaker so no need for a speaker identification line
        if (
            tr_segment["end"]
            <= clean_dz_segment_list[current_dz_segment_list_index]["end"]
        ):
            seg_content = tr_segment["text"]
            transcribed_text_lines_list.append(f"{seg_content}")

        # Segment falls outside the current speaker we need a new speaker identification line
        else:
            # If we are not on the last speaker so proceed as normal
            if current_dz_segment_list_index != (len(clean_dz_segment_list) - 1):
                # Point at the new speaker
                current_dz_segment_list_index += 1

                # Get the data to add to the speaker and timestamp line
                new_speaker = clean_dz_segment_list[current_dz_segment_list_index][
                    "label"
                ]
                new_speaker_start_time = clean_dz_segment_list[
                    current_dz_segment_list_index
                ]["start"]
                new_speaker_start_time = f"{float(new_speaker_start_time):.0f}"  # Format seconds to whole numbers
                new_speaker_start_time = str(
                    datetime.timedelta(seconds=float(new_speaker_start_time))
                )  # Format as H:M:S

                # Add the speaker and timestamp line
                transcribed_text_lines_list.append(
                    f"\n\n{new_speaker} [{new_speaker_start_time}]\n"
                )

                # Add the transcribed content
                seg_content = tr_segment["text"]
                transcribed_text_lines_list.append(f"{seg_content}")

            # We are on the last speaker so no more new speaker lines
            else:
                seg_content = tr_segment["text"]
                transcribed_text_lines_list.append(f"{seg_content}")

    # Post process the list of segments and speaker lines into a single string and clean
    transcribed_text = "".join(transcribed_text_lines_list)
    transcribed_text = transcribed_text.replace("\n ", "\n").lstrip()

    # Create text file
    text_file_path = Path(file_parent).joinpath(f"{file_name}.txt")
    file = open(text_file_path, "w")
    file.write(transcribed_text)

    # Delete the mp3 and wav
    os.remove(mp3_audio_file_path)
    os.remove(wav_audio_file_path)