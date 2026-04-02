"""
HTTP device agent for the Capstone dashboard (Raspberry Pi / face-detection node).

Endpoints:
  GET  /status
  POST /start
  POST /stop
  GET  /video_feed
  GET  /logs
"""

import glob
import logging

from flask import Flask, Response, jsonify
from flask_cors import CORS

import config
from stream_runner import PipelineStreamRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

runner = PipelineStreamRunner()


@app.get("/status")
def status():
    payload = {
        "online":  True,
        "running": runner.is_running(),
        "name":    config.DEVICE_NAME,
    }
    if runner.last_error:
        payload["error"] = runner.last_error
    return jsonify(payload)


@app.post("/start")
def start():
    success, message = runner.start()
    return jsonify({"success": success, "message": message}), (200 if success else 409)


@app.post("/stop")
def stop():
    success, message = runner.stop()
    return jsonify({"success": success, "message": message}), (200 if success else 409)


@app.get("/video_feed")
def video_feed():
    return Response(
        runner.mjpeg_chunks(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/logs")
def logs():
    files = sorted(glob.glob(str(config.ROOT_DIR / "logs" / "*.jsonl")))
    if not files:
        return jsonify({"logs": []})
    with open(files[-1]) as f:
        lines = f.readlines()[-20:]
    return jsonify({"logs": lines})


if __name__ == "__main__":
    log.info("Starting agent '%s' on port %d", config.DEVICE_NAME, config.PORT)
    app.run(host="0.0.0.0", port=config.PORT, threaded=True)
