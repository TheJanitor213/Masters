import wave
import pylab

from tqdm import tqdm
import os

from pydub.silence import split_on_silence
from pydub import AudioSegment

# Set paths to input and output data
INPUT_DIR = "raw_files\\"
OUTPUT_DIR = "new_data_by_subject\\"

# Print names of 10 WAV files from the input path


parent_list = [
    f"{subdir}\{file}"
    for subdir, dirs, files in tqdm(os.walk(INPUT_DIR))
    for file in files
    if file.endswith(".wav")
]
for i in range(10):
    print(parent_list[i])

# Utility function to get sound and frame rate info
def get_wav_info(wav_file):
    wav = wave.open(wav_file, "r")
    frames = wav.readframes(-1)
    sound_info = pylab.frombuffer(frames, "int16")
    frame_rate = wav.getframerate()
    wav.close()
    return sound_info, frame_rate





def match_target_amplitude(aChunk, target_dBFS):
    change_in_dBFS = target_dBFS - aChunk.dBFS
    return aChunk.apply_gain(change_in_dBFS)


counter = 0
for filename in parent_list:
    print(filename)
    if "wav" in filename:
        file_path = filename
        file_stem = filename.split("\\")[2]
        type = filename.split("\\")[1]
        target_dir = f"{file_stem[11]}"
        subject = file_stem[13:].replace(".wav", "")
        new_path = f"{type}\\{subject}"
        dist_dir = os.path.join(os.path.join(OUTPUT_DIR, new_path), target_dir)
        song = AudioSegment.from_wav(filename)

        dBFS = song.dBFS
        chunks = split_on_silence(
            song,
            min_silence_len=1000,
            # anything under -16 dBFS is considered silence
            silence_thresh=dBFS - 16,
            # keep 200 ms of leading/trailing silence
            keep_silence=200,
        )
        for i, chunk in enumerate(chunks):
            if i > 9:
                break
            silence_chunk = AudioSegment.silent(duration=200)
            audio_chunk = silence_chunk + chunk + silence_chunk
            normalized_chunk = match_target_amplitude(audio_chunk, -20.0)
            normalized_chunk = normalized_chunk.set_frame_rate(44100)
            file_dist_path = os.path.join(dist_dir, str(counter))
            print(file_dist_path)
            if not os.path.exists(file_dist_path + ".wav"):

                if not os.path.exists(dist_dir):

                    os.makedirs(dist_dir)
                normalized_chunk.export(file_dist_path+".wav", format="wav")
            counter += 1
