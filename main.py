from yt_dlp import YoutubeDL
import os
from pytube import YouTube
from anthropic import Anthropic
from dotenv import load_dotenv
import cv2
import base64
import numpy as np
import pytube as pt
import whisper
import time





print("🚀 Starting the Video Summarization Process...")
print()
print("🔑 Loading Anthropic API key...")

anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
client = Anthropic(api_key=anthropic_api_key)

ydl_opts = {
    'outtmpl': 'out/final.%(ext)s'
}
url = 'https://www.youtube.com/watch?v=W86cTIoMv2U'
print()
print("🌐 Downloading video from YouTube...")

with YoutubeDL(ydl_opts) as ydl:
    ydl.download([url])

print()
print("📽️ Extracting frames from the video...")

video = cv2.VideoCapture("out/final.mp4")
base64Frames = []
while video.isOpened():
    success, frame = video.read()
    if not success:
        break
    _, buffer = cv2.imencode(".jpg", frame)
    base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

video.release()

print()
print(f"📊 {len(base64Frames)} frames extracted.")

def create_frame_data(frames, fps=30):
    frame_data = []
    interval_duration = 2  # Duration of each interval in seconds
    frames_per_interval = interval_duration * fps
    time = 0

    for i in range(0, len(frames), frames_per_interval):
        frame_data.append((frames[i], f"Frame at {time} seconds into the video: "))
        time += 2

    return frame_data

print()

def generate_captions(url):

    print("🎵 Extracting audio from the video...")
    yt = pt.YouTube(url)
    stream = yt.streams.filter(only_audio=True)[0]
    stream.download(filename="out/final.mp3")
    print()
    print("🗣️ Generating captions using Whisper...")
    model = whisper.load_model("large")

    result = model.transcribe("out/final.mp3")
    segments = result["segments"]

    final_text = ""
    for segment in segments:
        start_time = segment["start"]
        end_time = segment["end"]
        text = segment["text"]
        final_text += f"{start_time:.2f} - {end_time:.2f}: {text}\n"

    return final_text

print()
print("📝 Generating captions...")

captions = generate_captions(url)

print()
print(captions)
print()
print("🎞️ Creating frame data...")

frame_data = create_frame_data(base64Frames, fps=30)

print()
print(f"📊 {len(frame_data)} frames processed.")

def send_to_anthropic(frames, transcript, previous_summary):
    
    SYSTEM_PROMPT = """You are an AI assistant designed to generate summaries of videos based on provided transcripts and frames. Your task is to analyze the given information and create a concise, coherent summary that captures the main points and key events of the video segment.

If no previous summary is provided, assume that the given frames and transcript represent the beginning of the video. If a previous summary is included, use it as context to understand the ongoing narrative and generate a summary that builds upon the existing information."""
    
    input_data = {
        "role": "user",
        "content": []
    }

    for frame, time_stamp in frames:
        input_data["content"].append({
            "type": "text",
            "text": time_stamp
        })

        input_data["content"].append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": frame
            }
        })

    if not previous_summary:
        input_data["content"].append({
            "type": "text",
            "text":  "These are the frames from the video along with the whole transcript, if i don't give you a previous summary it means these are the first frames of the video. Please use the information you have to generate the summary.  \n\n<Transcript>" + transcript + "</Transcript>"
        })
    else:
        input_data["content"].append({
            "type": "text",
            "text":  "These are the frames from the video along with the whole transcript, if i don't give you a previous summary it means these are the first frames of the video. Please use the information you have to generate the summary.  \n\n<Transcript>" + transcript + "</Transcript> \n\n <previous summary>" + previous_summary + "</previous summary> "
        })
    print()
    print("🧠 Sending data to Anthropic for summary generation...")
    response = client.messages.create(
        model = "claude-3-haiku-20240307",
        max_tokens=250,
        temperature=0,
        system=SYSTEM_PROMPT,
        messages = [input_data]
    )

    return response.content[0].text

group_size = 10
num_groups = (len(frame_data) + group_size - 1) // group_size

print()
print(f"📈 We will send {num_groups} API calls to the LLM.")

previous_summary = None

for i in range(num_groups):
    start_index = i * group_size
    print(f"🎬 Starting with frame at position {start_index}")
    print()

    end_index = min((i + 1) * group_size, len(frame_data))
    print()
    print(f"🎬 Ending with frame at position {end_index}")

    group_frames = frame_data[start_index:end_index]

    if previous_summary is None:
        summary = send_to_anthropic(group_frames, captions, None)
    else:
        summary = send_to_anthropic(group_frames, captions, previous_summary)
    previous_summary = summary

    print()
    print(f"📝 Summary for group {i + 1}:")
    print("=" * 40)
    print(summary)
    print("=" * 40)
    print()
    time.sleep(1)

print("✅ Video summarization complete!")