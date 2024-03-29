import logging
from pathlib import Path


from toolz import merge
from flask import Flask, jsonify, request
from flask_cors import CORS


from model import model_load, predict, MODEL


logging.basicConfig(
    filename="{}".format(Path.home() / "logs" / "predictor.log"),
    format="%(asctime)s == PILLITUP == PREDICTOR == [%(levelname)-8s] %(message)s",
    level=logging.DEBUG,
)

app = Flask("pill-it-up-cloud-predictor")
CORS(app)

model = model_load()


@app.route("/predict", methods=["GET", "POST"])
def prediction():
    logging.debug("Received prediction request.")

    img = request.files["media"]
    predicted_pill, all_probs = predict(model, img.stream)

    logging.debug("Found {}.".format(predicted_pill))
    ret_dict = merge({"medication": str(predicted_pill).replace("_", " ")}, {"all_probs": all_probs})
    return jsonify(ret_dict)


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
