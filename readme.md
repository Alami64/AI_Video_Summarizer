# Video Summarization with Anthropic AI

This Python script performs video summarization using the Anthropic AI API. It downloads a video from YouTube, extracts frames, generates captions using Whisper, and sends the frames and captions to the Anthropic AI for summary generation.

## Prerequisites

Before running the script, make sure you have the following dependencies installed:

* yt_dlp
* pytube
* anthropic
* python-dotenv
* opencv-python
* numpy
* whisper

You also need to have an Anthropic API key stored in a .env file.

you also need to have ffmpeg installed, if you have a mac, then just doing 'brew install ffmpeg' should work.

## Usage

1. Set the url variable to the YouTube video URL you want to summarize.
2. Run the script using the following command:

```bash
python video_summarization.py
```

3. The script will download the video, extract frames, generate captions, and send the data to the Anthropic AI for summary generation.
4. The generated summaries will be displayed in the console.

## Configuration

* The ydl_opts variable specifies the output template for the downloaded video.
* The interval_duration variable determines the duration of each interval in seconds when creating frame data.
* The group_size variable specifies the number of frames to send in each API call to the Anthropic AI.



