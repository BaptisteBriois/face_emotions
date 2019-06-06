from flask import Flask
from flask import Response

app = Flask(__name__)

@app.route("/upload", methods=["POST"])
def upload():
    imagefile = flask.request.files.get('imagefile', '')
    print (imagefile)
    resp = Response(imagefile, status=200, mimetype='application/json')
    return resp
