"""
Генерация вертикального РИЧ-баннера через OpenRouter (Gemini Image).

Кладите 3–10 изображений (jpg/png/jfif) в папку «входные», затем:
    python generate_rich.py
или:
    py -3 generate_rich.py

Ключ API: переменная окружения OPENROUTER_API_KEY или файл .env рядом со скриптом.

При ошибке SSL (корпоративная сеть): задайте SSL_CERT_FILE=путь\\к\\корневому.pem
или временно OPENROUTER_INSECURE_SSL=1 (только для отладки).
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import ssl
import sys
import time
from pathlib import Path

import certifi
import requests
from dotenv import load_dotenv
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "входные"
OUTPUT_DIR = ROOT / "выходные"

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "google/gemini-3.1-flash-image-preview"

MAX_BYTES_FILE = 5 * 1024 * 1024
MAX_TOTAL_DECODED = 16 * 1024 * 1024

EXT_TO_MIME = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".jfif": "image/jpeg",
    ".png": "image/png",
}

TIERS = [
    {"max_dim": 1024, "q": 76},
    {"max_dim": 800, "q": 72},
    {"max_dim": 640, "q": 68},
    {"max_dim": 512, "q": 64},
    {"max_dim": 400, "q": 60},
    {"max_dim": 320, "q": 56},
    {"max_dim": 256, "q": 52},
    {"max_dim": 192, "q": 48},
]


def mime_from_name(filename: str) -> str | None:
    return EXT_TO_MIME.get(Path(filename).suffix.lower())


def image_to_rgb_jpeg_bytes(raw: bytes, max_dim: int, quality: int) -> bytes:
    im = Image.open(io.BytesIO(raw))
    im = ImageOps.exif_transpose(im)

    if im.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", im.size, (255, 255, 255))
        if im.mode == "RGBA":
            bg.paste(im, mask=im.split()[3])
        else:
            bg.paste(im, mask=im.split()[1])
        im = bg
    elif im.mode == "P":
        if "transparency" in im.info:
            im_rgba = im.convert("RGBA")
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im_rgba, mask=im_rgba.split()[3])
            im = bg
        else:
            im = im.convert("RGB")
    elif im.mode != "RGB":
        im = im.convert("RGB")

    im.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def compress_all(buffers_meta: list[dict], tier: dict) -> tuple[list[bytes], int]:
    out: list[bytes] = []
    total = 0
    md = tier["max_dim"]
    q = tier["q"]
    for bm in buffers_meta:
        b = image_to_rgb_jpeg_bytes(bm["buffer"], md, q)
        out.append(b)
        total += len(b)
    return out, total


def shrink_buffers(buffers_meta: list[dict]) -> list[bytes]:
    for tier in TIERS:
        out, total = compress_all(buffers_meta, tier)
        if total <= MAX_TOTAL_DECODED:
            return out

    for md in range(176, 79, -16):
        q = max(36, 44 - (176 - md) // 24)
        out, total = compress_all(buffers_meta, {"max_dim": md, "q": q})
        if total <= MAX_TOTAL_DECODED:
            return out

    out, total = compress_all(buffers_meta, {"max_dim": 64, "q": 32})
    if total <= MAX_TOTAL_DECODED:
        return out

    raise RuntimeError(
        "Не удалось уложить изображения в лимит API (~30 МБ после base64). "
        "Уменьшите файлы или их число."
    )


def build_prompt() -> str:
    return "\n".join(
        [
            "Ты создаёшь один вертикальный РИЧ-баннер для маркетплейса: единое полотно визуального лендинга, не набор случайно склеенных картинок.",
            "Используй переданные изображения как референсы товара и стиля. Нумерация сверху вниз: изображение 1 … изображение N.",
            "Текст на баннере разрешён ТОЛЬКО из того, что реально читается на этих фото. Не придумывай состав, цифры, обещания и характеристики, если их нет на изображениях.",
            "Структура сверху вниз: Hero → краткие преимущества → при необходимости раскрытие → свойства/материалы (если видно на фото) → ассортимент только если на фото явно несколько SKU.",
            "Единая палитра и аккуратная типографика, читаемость.",
            "Сгенерируй одно итоговое изображение (вертикальный баннер).",
        ]
    )


def requests_verify_bundle() -> bool | str | ssl.SSLContext:
    """
    Аргумент requests(..., verify=...).

    Приоритет:
    1) OPENROUTER_INSECURE_SSL=1 — без проверки (небезопасно).
    2) SSL_CERT_FILE / REQUESTS_CA_BUNDLE — свой PEM.
    3) SSL_USE_CERTIFI=1 — только пакет certifi.
    4) Иначе — truststore: доверие к хранилищу Windows/macOS/Linux (корпоративные CA часто уже там).
    5) Нет truststore — certifi.
    """
    flag = os.environ.get("OPENROUTER_INSECURE_SSL", "").strip().lower()
    if flag in ("1", "true", "yes", "on"):
        print(
            "ВНИМАНИЕ: проверка TLS отключена (OPENROUTER_INSECURE_SSL).",
            file=sys.stderr,
        )
        return False
    ca = os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE")
    if ca:
        p = Path(ca)
        if p.is_file():
            return str(p.resolve())
        print(f"Файл сертификатов не найден: {ca}", file=sys.stderr)
    if os.environ.get("SSL_USE_CERTIFI", "").strip().lower() in ("1", "true", "yes", "on"):
        return certifi.where()
    try:
        import truststore

        return truststore.ssl_context()
    except ImportError:
        return certifi.where()


def extract_image_from_response(data: dict) -> tuple[bytes, str]:
    msg = (data.get("choices") or [{}])[0].get("message") or {}
    images = msg.get("images") or []
    if not images:
        raise ValueError("В ответе API нет поля images: " + json.dumps(data, ensure_ascii=False)[:800])

    url = (images[0].get("image_url") or {}).get("url") or ""
    m = re.match(r"^data:([^;]+);base64,(.+)$", url, re.DOTALL)
    if not m:
        raise ValueError("Неожиданный формат картинки в ответе.")

    mime = m.group(1)
    raw_b64 = m.group(2)
    return base64.b64decode(raw_b64), mime


def main() -> int:
    load_dotenv(ROOT / ".env")

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        print("Задайте OPENROUTER_API_KEY в файле .env или в переменной окружения.", file=sys.stderr)
        return 1

    if not INPUT_DIR.is_dir():
        INPUT_DIR.mkdir(parents=True, exist_ok=True)
        print(
            'Создана папка «входные». Положите туда 3–10 изображений и запустите снова.',
            file=sys.stderr,
        )
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    names = sorted(
        [p.name for p in INPUT_DIR.iterdir() if p.is_file() and mime_from_name(p.name)],
        key=lambda x: x.lower(),
    )

    if len(names) < 3 or len(names) > 10:
        print(
            f"Нужно от 3 до 10 изображений в «входные». Подходящих файлов: {len(names)}.",
            file=sys.stderr,
        )
        return 1

    buffers_meta: list[dict] = []
    for name in names:
        fp = INPUT_DIR / name
        buf = fp.read_bytes()
        if len(buf) > MAX_BYTES_FILE:
            print(f"Файл слишком большой (>5 МБ): {name}", file=sys.stderr)
            return 1
        mime = mime_from_name(name)
        assert mime
        buffers_meta.append({"buffer": buf, "mime": mime, "name": name})

    print("Сжатие изображений под лимит API…", file=sys.stderr)
    try:
        final_buffers = shrink_buffers(buffers_meta)
    except Exception as e:
        print(str(e), file=sys.stderr)
        return 1

    content: list[dict] = [{"type": "text", "text": build_prompt()}]
    for buf in final_buffers:
        b64 = base64.b64encode(buf).decode("ascii")
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            }
        )

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": content}],
        "modalities": ["image", "text"],
        "image_config": {"aspect_ratio": "1:4", "image_size": "2K"},
    }

    verify = requests_verify_bundle()

    print("Запрос к OpenRouter…", file=sys.stderr)
    try:
        r = requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/rich-content-cli",
                "X-Title": "Rich Content Python",
            },
            json=payload,
            timeout=300,
            verify=verify,
        )
    except requests.exceptions.SSLError as e:
        print(
            "Ошибка TLS при обращении к openrouter.ai.\n"
            "  • Выполните: pip install -r requirements.txt (нужны пакеты truststore, certifi).\n"
            "  • По умолчанию используется хранилище Windows — корпоративный CA должен быть в «Доверенные корневые».\n"
            "  • Либо укажите PEM с корнем прокси в .env: SSL_CERT_FILE=C:\\path\\to\\corp.pem\n"
            "  • Либо временно: OPENROUTER_INSECURE_SSL=1 (небезопасно)\n"
            f"Исходное сообщение: {e}",
            file=sys.stderr,
        )
        return 1

    try:
        data = r.json()
    except json.JSONDecodeError:
        print("Ответ не JSON:", r.text[:500], file=sys.stderr)
        return 1

    if not r.ok:
        print("Ошибка OpenRouter:", r.status_code, json.dumps(data, ensure_ascii=False), file=sys.stderr)
        return 1

    try:
        out_bytes, out_mime = extract_image_from_response(data)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1

    ext = "png" if "png" in out_mime.lower() else "jpg"
    out_path = OUTPUT_DIR / f"rich-{int(time.time() * 1000)}.{ext}"
    out_path.write_bytes(out_bytes)
    print(out_path.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
