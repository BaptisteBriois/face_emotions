from flask import Flask
from flask import Response
from video_emotion_color_demo import main

app = Flask(__name__)

@app.route("/upload", methods=["POST"])
def upload():
    imagefile = flask.request.files.get('imagefile', '')
    print (imagefile)
    resp = Response(main(imagefile), status=200, mimetype='application/json')
    return resp
