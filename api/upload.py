from flask import Flask
from flask import Response
from flask import request
from video_emotion_color_demo import main

from flask_cors import CORS, cross_origin
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

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
        resp = Response(main(imagefile), status=200, mimetype='application/json')
        return resp

app.run(port=9000)