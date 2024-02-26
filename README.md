# Transcribe Video

A script that takes the path to a video file and diarises it using Simple Diariser and transcribes it using Whisper from OpenAI. It then places a summary at the top using ChatGPT.

## Usage:

transcribe_v_ideo.py [-h] [-n NUMBER_SPEAKERS] file_path [file_path ...]

## Positional Arguments:

file_path

A full or relative path to a media file, several media files or a directory of media files to transcribe. Typically these will be videos or mp3s.


## Optional Arguments:

-n NUMBER_SPEAKERS, --number_speakers NUMBER_SPEAKERS

The number of speakers in the video. Default = 2

-h, --help

Show this help message and exit
