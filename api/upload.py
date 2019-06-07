from flask import Flask
from flask import Response
from flask import request
from video_emotion_color_demo import main
from flask_cors import CORS, cross_origin
import tensorflow as tf
import time
import json
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

graph = tf.get_default_graph()

@app.route("/", methods=["POST"])
def index():
    global graph
    with graph.as_default():
        json_data = request.get_json(force=True)
        imagefile = json_data['screenshot']
        data = main(imagefile)
        print (data)
        return Response(json.dumps({'body': main(imagefile)}), status=200, mimetype='application/json')

app.run(port=9000)