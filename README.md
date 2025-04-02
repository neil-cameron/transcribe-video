# Transcribe Video

A script that takes the path to a video file and diarises it using Pyannote and transcribes it using Faster Whisper from OpenAI. It then places a summary at the top using ChatGPT.

## Usage:

transcribe_video.py [-h] file_path [file_path ...]

## Positional Arguments:

file_path

A full or relative path to a media file, several media files or a directory of media files to transcribe. Typically these will be videos or mp3s.


## Optional Arguments:

-h, --help

Show this help message and exit


## Additional Requirements:

You will need a file in the root of this project called `config.py` with the below contents:

```
authorization = "open ai api key goes here"  # openai
hf_authorization = "huggig face token for pyannote goes here"
```