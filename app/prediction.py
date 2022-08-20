
from flask import Blueprint, request
import boto3
from libsvm.svmutil import *
import numpy as np
import pickle
import logging as logger
import json
from python_speech_features import mfcc
prediction = Blueprint("prediction", __name__, url_prefix="/")



@prediction.before_request
def before():
    logger.info(request.url)
    logger.debug(request.__dict__)
    logger.debug(request.headers)


@prediction.after_request
def after(response):
    logger.debug(response.status)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

    
def pad_audio(data, fs, T=3):
    # Calculate target number of samples
    N_tar = int(fs * T)
    # Calculate number of zero samples to append
    shape = data.shape
    # Create the target shape
    N_pad = N_tar - shape[0]
    # print("Padding with %s seconds of silence" % str(N_pad/fs) )
    shape = (N_pad,) + shape[1:]
    # Stack only if there is something to append
    if shape[0] > 0:
        if len(shape) > 1:
            return np.vstack((np.zeros(shape), data))
        else:
            return np.hstack((np.zeros(shape), data))
    else:
        return data

def load_audio(audio_file):
    from scipy.io import wavfile
    samplerate, data = wavfile.read(audio_file)
    return pad_audio(data, samplerate)

def extract_features(audio_data):
    data = mfcc(audio_data, samplerate=22050, nfft=2048, winfunc=np.hamming)
    return np.array(data)

from .common import split_sound
@prediction.route("/prediction",methods=["POST"])
def predict():
    try:

        logger.info("Fetching pre-trained model from S3")
        s3 = boto3.client("s3")
        bucket = "zappa-ax0b7nbq7"
        s3 = boto3.resource("s3")
        model_file_name = '/tmp/sklearn_model.sav'
        s3.meta.client.download_file(bucket, "sklearn_model.sav", model_file_name)

        from string import ascii_uppercase
        mapping = {float(k):v for k,v in enumerate(ascii_uppercase)}
        uploaded_file = request.files["audio_data"]
        saved_file = "/tmp/" + uploaded_file.filename + ".wav"
        uploaded_file.save(saved_file)
        
        files = split_sound(saved_file)
        all_features = []
        for file in files:
            logger.debug(file)
            audio_data = load_audio(file)
            features = extract_features(audio_data)
            all_features.append(features.flatten())

        logger.debug(all_features)
        # Calculate prediction
        try:
            #values = {ind:v for ind, v in enumerate(features.flatten().tolist())}

            model = pickle.load(open(model_file_name, 'rb'))
            #prediction = loaded_classifier.predict(features.flatten().reshape(1,-1))
            prediction = model.predict(all_features)
            word = ''.join([mapping[k] for k in prediction])
            prediction_payload = {
                "word": word
            }
            logger.info(word)
            return {    
                "statusCode": 200,
                "body":     
                    {
                        "message": "Success",
                        "prediction": word
                    }
                ,
            }

        except Exception as e:
            logger.error('Unhandled error: {}'.format(e))
            return {
                "statusCode": 500,
                "body": json.dumps(
                    {
                        "message": "Unhandled error",
                    }
                ),
            }
    except Exception as e:
        logger.exception(e)
        return "Error", 500
