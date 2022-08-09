
from flask import Flask

from app.processing import processing
from app.prediction import prediction
app = Flask(__name__)
app.register_blueprint(processing)
app.register_blueprint(prediction)
