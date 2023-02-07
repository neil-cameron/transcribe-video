import argparse
import whisper
from pathlib import Path
import subprocess
import os
import datetime

# from pyannote.audio import Pipeline
import soundfile as sf
from simple_diarizer.diarizer import (
    Diarizer,
)  # https://github.com/cvqluu/simple_diarizer

parser = argparse.ArgumentParser(
    prog="Transcribe Video",
    description="This program takes the path to a video file and transcribes it using Whisper from OpenAI",
)

parser.add_argument("file_path", nargs="+")
args = parser.parse_args()

model = whisper.load_model("base.en")
# pipeline = Pipeline.from_pretrained(
#     "pyannote/speaker-diarization@2.1",
#     use_auth_token="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",
# )
diar = Diarizer(
    embed_model="xvec",  # 'xvec' and 'ecapa' supported
    cluster_method="sc",  # 'ahc' and 'sc' supported
)

for file_path_item in args.file_path:
    file_name = Path(file_path_item).stem
    file_parent = Path(file_path_item).parents[0]
    audio_file_path = Path(file_parent).joinpath(f"{file_name}.mp3")

    # Generate an mp3
    command = f"ffmpeg -i '{file_path_item}' '{audio_file_path}'"
    subprocess.call(command, shell=True)

    # Diarization
    # audio_file_dict_for_diarization = {
    #     "uri": "complete_audio_file",
    #     "audio": audio_file_path,
    # }
    # diarization = pipeline(audio_file_dict_for_diarization)
    # print(diarization)
    diarization = diar.diarize(audio_file_path, num_speakers=2)
    # print(diarization)

    clean_segment_list = []
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
            clean_segment_list.append(dict_to_add)
            current_speaker = segment["label"]
            current_speaker_start = segment["start"]
    print(clean_segment_list)

    # # Transcribe
    # transcribed_text_segment_dict_list = model.transcribe(str(audio_file_path))[
    #     "segments"
    # ]

    # # Build a new text file including time stamps line by line
    # transcribed_text_lines_list = []
    # segment_counter = 0
    # for segment in transcribed_text_segment_dict_list:
    #     segment_counter += 1
    #     seg_start = segment["start"]
    #     seg_start = f"{float(seg_start):.0f}"  # Format seconds to whole numbers
    #     converted_seg_start = str(
    #         datetime.timedelta(seconds=float(seg_start))
    #     )  # Format as H:M:S
    #     if segment_counter % 15 == 0:  # Only print out timestamps every x segments
    #         transcribed_text_lines_list.append(
    #             f"\n\n[Timestamp: {converted_seg_start}]\n"
    #         )
    #     seg_content = segment["text"]
    #     transcribed_text_lines_list.append(f"{seg_content}")
    # transcribed_text = "".join(transcribed_text_lines_list)
    # transcribed_text = transcribed_text.replace("\n ", "\n").lstrip()

    # # Create text file
    # text_file_path = Path(file_parent).joinpath(f"{file_name}.txt")
    # file = open(text_file_path, "w")
    # file.write(transcribed_text)

    # # Delete the mp3
    # os.remove(audio_file_path)