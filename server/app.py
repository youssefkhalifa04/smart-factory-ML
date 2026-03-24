import sys
from pathlib import Path

import flask
from flask import request, jsonify
from flask_cors import CORS

try:
    from model.model import run
except ModuleNotFoundError:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from model.model import run

app = flask.Flask(__name__)
app.config["DEBUG"] = True
CORS(app, resources={r"/api/*": {"origins": "*"}})


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    factory_id = data.get("factory_id")
    degree = data.get("degree", 2)

    if not factory_id:
        return jsonify({"status": "error", "message": "factory_id is required"}), 400

    try:
        degree = int(degree)
    except (TypeError, ValueError):
        return jsonify({"status": "error", "message": "degree must be an integer"}), 400
    
    try:
        predictions = run(factory_id=factory_id, degree=degree)
        return jsonify({"status": "success", "prediction": predictions[0] if predictions else None})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})
    

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
