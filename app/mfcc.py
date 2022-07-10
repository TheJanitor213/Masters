import python_speech_features as mfcc
import numpy as np
from sklearn import preprocessing
import logging 

def extract_features(audio,rate):
    """extract 20 dim mfcc features from an audio, performs CMS and combines delta to make it 40 dim feature vector"""

    mfcc_feature = mfcc.mfcc(audio,rate, 0.025, 0.01,20,nfft = 1200, appendEnergy = True)
    mfcc_feature = preprocessing.scale(mfcc_feature)
    logging.debug(mfcc_feature)
    return mfcc_feature