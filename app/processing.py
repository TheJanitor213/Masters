from flask import request, Blueprint
import logging
import boto3

from pydub.silence import split_on_silence
from pydub import AudioSegment

FORMAT = "%(asctime)-15s %(clientip)s %(user)-8s %(message)s"
logger = logging.getLogger("test")
logger.setLevel(logging.DEBUG)
processing = Blueprint("index", __name__, url_prefix="/")



@processing.before_request
def before():
    logger.info(request.url)
    logger.debug(request.__dict__)
    logger.debug(request.headers)


@processing.after_request
def after(response):
    logger.debug(response.status)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

@processing.route("/")
def index():
    return "Hello World!"

@processing.route("/upload", methods=["POST"])
def upload_file():
    try:
        uploaded_file = request.files["audio_data"]
        if uploaded_file.filename != "":
            uploaded_file.save("/tmp/" + uploaded_file.filename + ".wav")
            s3 = boto3.client("s3")
            s3.upload_file(
                "/tmp/" + uploaded_file.filename + ".wav",
                "datasets-masters-2020",
                uploaded_file.filename + ".wav",
            )
        return "Success", 200
    except Exception as e:
        logger.exception(e)
        return "Error", 500



def match_target_amplitude(aChunk, target_dBFS):
    change_in_dBFS = target_dBFS - aChunk.dBFS
    return aChunk.apply_gain(change_in_dBFS)


def process_file(event, context):
    try:
        FFMPEG_STATIC = "/var/task/ffmpeg"

        AudioSegment.converter = "/opt/python/ffmpeg"

        bucket = event["Records"][0]["s3"]["bucket"]["name"]
        s3_client = boto3.client("s3")
        key = event["Records"][0]["s3"]["object"]["key"]
        if key.split("/")[0] != "splits":
            # Get the bytes from S3
            file_loc = "/tmp/" + key
            # Download this file to writable tmp space.
            logger.debug(file_loc)
            logger.debug(key)
            logger.debug(bucket)
            s3_client.download_file(bucket, key, file_loc)
            song = AudioSegment.from_wav(file_loc)

            dBFS = song.dBFS
            chunks = split_on_silence(
                song,
                min_silence_len=1000,
                # anything under -16 dBFS is considered silence
                silence_thresh=dBFS - 16,
                # keep 200 ms of leading/trailing silence
                keep_silence=200,
            )
            logger.debug(chunks)
            for i, chunk in enumerate(chunks):
                silence_chunk = AudioSegment.silent(duration=200)
                audio_chunk = silence_chunk + chunk + silence_chunk
                normalized_chunk = match_target_amplitude(audio_chunk, -20.0)
                logger.debug("Exporting chunk{0}.wav.".format(i))
                normalized_chunk = normalized_chunk.set_frame_rate(44100)
                normalized_chunk.export("/tmp/chunk{0}.wav".format(i), format="wav")
                s3_client.upload_file(
                    "/tmp/chunk{0}.wav".format(i),
                    "datasets-masters-2020",
                    "splits/{0}/chunk_{1}.wav".format(key.split(".")[0], i),
                )
            return
        else:
            logger.debug("Nothing to do here")
            return

    except Exception as e:
        logger.exception(e)
        return "Error", 500
