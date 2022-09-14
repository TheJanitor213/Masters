import os
from scipy.io import wavfile
from tqdm import tqdm
import os
from pydub.silence import split_on_silence
from pydub import AudioSegment


def match_target_amplitude(aChunk, target_dBFS):
    change_in_dBFS = target_dBFS - aChunk.dBFS
    return aChunk.apply_gain(change_in_dBFS)


def load_audio():
    rootdir = "../temp/"
    letters = {}
    print("Loading audio...")
    file_count = sum(len(files) for _, _, files in os.walk(rootdir))
    print(file_count)
    counter = 1301
    with tqdm(total=file_count) as pbar:
        for subdir, dirs, files in tqdm(os.walk(rootdir)):
            print(subdir, files)
            for file in files:
                pbar.update(1)
                song = AudioSegment.from_wav(subdir+file)

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
                    silence_chunk = AudioSegment.silent(duration=200)
                    audio_chunk = silence_chunk + chunk + silence_chunk
                    normalized_chunk = match_target_amplitude(audio_chunk, -20.0)
                    normalized_chunk = normalized_chunk.set_frame_rate(44000)
                    letter = file.split("-")[3]
                    subject = int(file.split("-")[5][0]) + 5 
                    try:
                        normalized_chunk.export(
                            f"data_by_subject/Subject - {subject}/{letter}/{counter}_.wav",
                            format="wav",
                        )
                    except:
                        os.makedirs(f"data_by_subject/Subject - {subject}/{letter}/")
                        normalized_chunk.export(
                            f"data_by_subject/Subject - {subject}/{letter}/{counter}_.wav",
                            format="wav",
                        )
                    counter +=1

    return letters

load_audio()