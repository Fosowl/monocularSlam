
import os
import sys
import yt_dlp as youtube_dl

OUTPUT_FOLDER_PATH = "./videos"

def confirm_download(full_path, min_bytes):
    if os.path.exists(full_path) == False:
        return False
    if os.path.getsize(full_path) < min_bytes:
        return False
    return True

def safe_remove(path):
    if os.path.exists(path):
        os.remove(path)

def get_yt_options(full_path, start_stamp, duration_stamp):
    return {
        'quiet': False,
        'format': 'best',
        'outtmpl': full_path,
        'noplaylist': True,
        'continue_dl': True,
        'postprocessor_args': ['-ss', start_stamp, '-t', duration_stamp]
    }
    
def download_clip(url: str, name: str, path_folder: str, timestamp="00:00:00", duration="00:01:00") -> bool:
    full_path = f'{path_folder}/{name}' 
    full_path_wav = f'{full_path}.mp4' 
    ydl_opts = get_yt_options(full_path,
                              timestamp,
                              duration)
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.cache.remove()
            info_dict = ydl.extract_info(url, download=False)
            if info_dict == None:
                print("empty info dict")
                return False
            try:
                duration = info_dict['duration']
            except:
                print("video is live stream")
                return False
            if duration > 36000:
                print("video too long for download")
                return False
            ydl.prepare_filename(info_dict)
            ydl.download([url])
        except Exception as e:
            print(f"Fatal error on download of {name} : {e}")
            return False
    if confirm_download(full_path_wav, 20000) == False:
        safe_remove(full_path_wav)
        return False
    return True

url = input("url :")
name = input("name :")
download_clip(url, name, "videos/", duration="00:02:00")