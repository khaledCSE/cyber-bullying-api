import pickle
import numpy as np
from flask import Flask, request, json, jsonify
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)

app.config['CORS_HEADERS'] = 'Content-Type'

@app.route("/", methods=["POST"])
@cross_origin()
def post_comment():
    # Taking JSON input
    jsonData = request.get_json(force=True)
    jsonStr = json.dumps(jsonData)
    data = json.loads(jsonStr)
    
    # vectorizer = pickle.load(open('./vectorizer.pkl', 'rb'))
    # model = pickle.load(open('./model.pkl', 'rb'))

    vectorizer = pickle.load(open('./vectorizer-ml.pkl', 'rb'))
    model = pickle.load(open('./model-ml.pkl', 'rb'))
    
    text = np.zeros(100)
    text = [data['comment']]
    text = vectorizer.transform(text)

    prediction = model.predict(text)

    val = np.int16(prediction)

    result = val.item()

    return jsonify(result = result)

if __name__ == "__main__":
    app.run(debug=True)