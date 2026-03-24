import cors 
import flask
from flask import request, jsonify
from model.model import run

app = flask.Flask(__name__)
app.config["DEBUG"] = True
cors = cors.CORS(app, resources={r"/api/*": {"origins": "*"}})


app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json()
    factory_id = data.get("factory_id")
    degree = data.get("degree", 2)
    
    try:
        run(factory_id=factory_id, degree=degree)
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})
    

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
