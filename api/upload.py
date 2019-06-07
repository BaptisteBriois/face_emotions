from flask import Flask
from flask import Response
from flask import request
from video_emotion_color_demo import main
from flask_cors import CORS, cross_origin
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route("/", methods=["POST"])
def index():
    json_data = request.get_json(force=True)
    imagefile = json_data['screenshot']
    resp = Response(main(imagefile), status=200, mimetype='application/json')
    return resp

app.run(port=9000)