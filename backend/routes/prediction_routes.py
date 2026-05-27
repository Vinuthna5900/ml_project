from flask import Blueprint, request, jsonify
from src.predict import predict
from database.logger import init_db, log_prediction, get_history

prediction_bp = Blueprint("prediction", __name__)

init_db()


@prediction_bp.route("/predict", methods=["POST"])
def get_prediction():
    try:
        data = request.get_json()

        prediction, confidence = predict(data)

        fraud_alert = bool(prediction == 1)

        status = "Eligible" if prediction == 1 else "Not Eligible"

        log_prediction(
            input_data=data,
            prediction=prediction,
            confidence=confidence,
            status=status,
            fraud_alert=int(fraud_alert)
        )

        return jsonify({
            "prediction": prediction,
            "confidence": round(confidence * 100, 2),
            "status": status,
            "fraud_alert": fraud_alert
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@prediction_bp.route("/history", methods=["GET"])
def history():
    rows = get_history()

    data = [
        {
            "timestamp": row[0],
            "prediction": row[1],
            "confidence": row[2],
            "status": row[3],
            "fraud_alert": row[4]
        }
        for row in rows
    ]

    return jsonify(data)