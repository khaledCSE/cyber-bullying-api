import pickle
import numpy as np
from flask import Flask, request, json, jsonify

app = Flask(__name__)

@app.route("/", methods=["POST"])
def post_comment():
    # Taking JSON input
    jsonData = request.get_json(force=True)
    jsonStr = json.dumps(jsonData)
    data = json.loads(jsonStr)
    
    vectorizer = pickle.load(open('./vectorizer.pkl', 'rb'))
    model = pickle.load(open('./model.pkl', 'rb'))
    
    text = np.zeros(100)
    text = [data['comment']]
    text = vectorizer.transform(text)

    prediction = model.predict(text)

    val = np.int16(prediction)

    result = val.item()

    return jsonify(result = result)

if __name__ == "__main__":
    app.run(debug=True)