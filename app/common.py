from pydub.silence import split_on_silence
from pydub import AudioSegment



def match_target_amplitude(aChunk, target_dBFS):
    change_in_dBFS = target_dBFS - aChunk.dBFS
    return aChunk.apply_gain(change_in_dBFS)


def split_sound(raw_audio_file): 
    FFMPEG_STATIC = "/var/task/ffmpeg"

    AudioSegment.converter = "/opt/python/ffmpeg"
    song = AudioSegment.from_wav(raw_audio_file)

    dBFS = song.dBFS
    chunks = split_on_silence(
        song,
        min_silence_len=1000,
        # anything under -16 dBFS is considered silence
        silence_thresh=dBFS - 16,
        # keep 200 ms of leading/trailing silence
        keep_silence=200,
    )
    files = []
    for i, chunk in enumerate(chunks):
        silence_chunk = AudioSegment.silent(duration=200)
        audio_chunk = silence_chunk + chunk + silence_chunk
        normalized_chunk = match_target_amplitude(audio_chunk, -20.0)

        normalized_chunk = normalized_chunk.set_frame_rate(44000)
        normalized_chunk.export("/tmp/chunk{0}.wav".format(i), format="wav")
        files.append("/tmp/chunk{0}.wav".format(i))
    return files