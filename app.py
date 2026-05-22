import os
import sys
import shutil
import torch
import yt_dlp
from yt_dlp.utils import download_range_func
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

def load_env():
    """Load local environment variables from .env if present."""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                if "=" in line and not line.strip().startswith("#"):
                    k, v = line.split("=", 1)
                    os.environ[k.strip()] = v.strip().strip("'\"")

load_env()

def download_video(url, out_name="test_video.mp4"):
    """Downloads the first 2 minutes of a video in low resolution using yt-dlp."""
    if not shutil.which("ffmpeg"):
        print("Error: FFmpeg is missing. Install Gyan.FFmpeg or verify system PATH.")
        sys.exit(1)
        
    print(f"Downloading first 2 minutes of {url}...")
    opts = {
        'format': 'worstvideo[ext=mp4]+worstaudio[ext=m4a]/worst',
        'merge_output_format': 'mp4',
        'outtmpl': out_name,
        'download_ranges': download_range_func(None, [(0, 120)]),
        'force_keyframes_at_cuts': True,
        'quiet': True,
        'overwrites': True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])
    print(f"Video slice saved as {out_name}")

def run_marlin(video_path):
    """Loads NemoStation/Marlin-2B and processes the video."""
    model_id = "NemoStation/Marlin-2B"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Select float precision dtype (bfloat16 for GPUs, float32 for CPU fallback)
    torch_dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else (torch.float16 if device == "cuda" else torch.float32)
    device_map = "auto" if device == "cuda" else None
    
    print(f"Loading {model_id} on {device}...")
    # Load model with memory-mapping disabled for robust CPU loading
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch_dtype, device_map=device_map, low_cpu_mem_usage=False
    )
    processor = AutoProcessor.from_pretrained(model_id)
    
    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": video_path, "max_pixels": 360 * 420, "nframes": 32},
            {"type": "text", "text": "Describe the events in this video with exact timestamps to generate chapters."}
        ]
    }]
    
    # Process inputs through Qwen chat template and extract vision metadata
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_in, video_in, video_kw = process_vision_info(messages, return_video_kwargs=True)
    
    # Prepare PyTorch model tensors
    inputs = processor(
        text=[text], images=image_in, videos=video_in, padding=True, return_tensors="pt", **video_kw
    ).to(device)
    
    print("Inferring video timestamps with Marlin-2B...")
    with torch.no_grad():
        gen_ids = model.generate(**inputs, max_new_tokens=512)
    
    # Slice user prompt tokens from generated output tokens and decode text response
    trimmed = [g[len(i):] for i, g in zip(inputs.input_ids, gen_ids)]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("url", nargs="?", default="https://www.youtube.com/watch?v=yoa4opGsvP4")
    args = parser.parse_args()
    
    download_video(args.url)
    chapters = run_marlin("test_video.mp4")
    
    with open("chapters.txt", "w", encoding="utf-8") as f:
        f.write(chapters)
    print(f"Chapters written to chapters.txt:\n{chapters}")

if __name__ == "__main__":
    main()
