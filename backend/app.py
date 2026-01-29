import os
from flask import Flask, render_template, request
from predict import predict_image, predict_video

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    result = ""

    if request.method == "POST":
        file = request.files.get("file")

        if file and file.filename != "":
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)

            if file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
                result = predict_image(file_path)

            elif file.filename.lower().endswith((".mp4", ".avi", ".mov")):
                result = predict_video(file_path)

            else:
                result = "Unsupported file format"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
