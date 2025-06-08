from flask import Flask, jsonify
import datetime
import random

app = Flask(__name__)

WINDOW_SIZE = 300
BASE_VALUE = 0.5
WIGGLE = 0.05

@app.route("/data", methods=["GET"])
def get_signal_data():
    now = datetime.datetime.utcnow()
    
    data = []
    for i in range(WINDOW_SIZE):
        value = BASE_VALUE + random.uniform(-WIGGLE, WIGGLE)
        timestamp = now - datetime.timedelta(seconds=WINDOW_SIZE - i)
        iso_timestamp = timestamp.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        data.append({
            "value": round(value, 6),
            "time": iso_timestamp
        })

    return jsonify(data)

if __name__ == "__main__":
    print("Starting mock signal server...")
    app.run(host="127.0.0.1", port=5000, debug=True)
