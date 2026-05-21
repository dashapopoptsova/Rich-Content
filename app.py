import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

from flask import Flask, render_template, request, redirect, url_for, send_file

from generate_rich import run_pipeline, mime_from_name, OUTPUT_DIR

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 60 * 1024 * 1024  # 60 МБ — до 10 файлов по 5 МБ + запас


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    files = request.files.getlist("images")

    if not (3 <= len(files) <= 20):
        return render_template(
            "index.html",
            error=f"Нужно от 3 до 20 изображений. Загружено: {len(files)}.",
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        input_paths: list[Path] = []

        for f in files:
            if not mime_from_name(f.filename):
                return render_template(
                    "index.html",
                    error=f"Неподдерживаемый формат: {f.filename}. Используйте JPG или PNG.",
                )
            safe_name = Path(f.filename).name
            dest = tmp / safe_name
            f.save(dest)
            input_paths.append(dest)

        try:
            out_path = run_pipeline(input_paths)
        except Exception as e:
            return render_template("index.html", error=str(e))

    return redirect(url_for("result", filename=out_path.name))


@app.route("/result/<filename>")
def result(filename):
    file_path = (OUTPUT_DIR / filename).resolve()
    if not str(file_path).startswith(str(OUTPUT_DIR.resolve())):
        return "Недопустимый путь", 400
    if not file_path.is_file():
        return "Файл не найден", 404
    return render_template("result.html", filename=filename)


@app.route("/preview/<filename>")
def preview(filename):
    file_path = (OUTPUT_DIR / filename).resolve()
    if not str(file_path).startswith(str(OUTPUT_DIR.resolve())):
        return "Недопустимый путь", 400
    if not file_path.is_file():
        return "Файл не найден", 404
    return send_file(file_path)


@app.route("/download/<filename>")
def download(filename):
    file_path = (OUTPUT_DIR / filename).resolve()
    if not str(file_path).startswith(str(OUTPUT_DIR.resolve())):
        return "Недопустимый путь", 400
    if not file_path.is_file():
        return "Файл не найден", 404
    return send_file(file_path, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
