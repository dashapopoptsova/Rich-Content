"""
Генерация вертикального РИЧ-баннера через OpenRouter (Gemini Image).
Один вызов API: анализ цвета + OCR + компоновка за один запрос.

Кладите 3–10 изображений (jpg/png/jfif) в папку «входные», затем:
    python generate_rich.py
или:
    py -3 generate_rich.py

Ключ API: переменная окружения OPENROUTER_API_KEY или файл .env рядом со скриптом.
Промпт: файл prompt.txt рядом со скриптом.

При ошибке SSL (корпоративная сеть): задайте SSL_CERT_FILE=путь\\к\\корневому.pem
или временно OPENROUTER_INSECURE_SSL=1 (только для отладки).
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
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
MODEL = os.environ.get("OPENROUTER_MODEL", "google/gemini-3.1-flash-image-preview")

MAX_BYTES_FILE = 5 * 1024 * 1024
MAX_TOTAL_DECODED = 4 * 1024 * 1024


MAX_RETRIES = 3
RETRY_DELAY = 20

EXT_TO_MIME = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".jfif": "image/jpeg",
    ".png": "image/png",
}

TIERS = [
    {"max_dim": 1024, "q": 76},
    {"max_dim": 800,  "q": 72},
    {"max_dim": 640,  "q": 68},
    {"max_dim": 512,  "q": 64},
    {"max_dim": 400,  "q": 60},
    {"max_dim": 320,  "q": 56},
    {"max_dim": 256,  "q": 52},
    {"max_dim": 192,  "q": 48},
]


# ─────────────────────────────────────────────────────────────
# Утилиты изображений
# ─────────────────────────────────────────────────────────────

def mime_from_name(filename: str) -> str | None:
    return EXT_TO_MIME.get(Path(filename).suffix.lower())


def image_to_rgb_jpeg_bytes(im: Image.Image, max_dim: int, quality: int) -> bytes:
    """Принимает уже открытый PIL.Image, возвращает сжатый JPEG-байты."""
    # Клонируем чтобы не мутировать оригинал
    im = im.copy()

    if im.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", im.size, (255, 255, 255))
        mask_channel = im.split()[3] if im.mode == "RGBA" else im.split()[1]
        bg.paste(im, mask=mask_channel)
        im = bg
    elif im.mode == "P":
        if "transparency" in im.info:
            im_rgba = im.convert("RGBA")
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im_rgba, mask=im_rgba.split()[3])
            im = bg
        else:
            im = im.convert("RGB")
    elif im.mode == "CMYK":
        im = im.convert("RGB")
    elif im.mode != "RGB":
        im = im.convert("RGB")

    im.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def load_images(buffers_meta: list[dict]) -> list[dict]:
    """Предварительно открывает все PIL.Image — один раз для всей компрессии."""
    result = []
    for bm in buffers_meta:
        try:
            raw_im = Image.open(io.BytesIO(bm["buffer"]))
            raw_im = ImageOps.exif_transpose(raw_im)
            raw_im.load()  # форсируем декодирование в память
        except Exception as e:
            raise RuntimeError(f"Не удалось открыть изображение «{bm['name']}»: {e}") from e
        result.append({**bm, "pil": raw_im})
    return result


def compress_all(images: list[dict], tier: dict) -> tuple[list[bytes], int]:
    out: list[bytes] = []
    total = 0
    for img in images:
        b = image_to_rgb_jpeg_bytes(img["pil"], tier["max_dim"], tier["q"])
        out.append(b)
        total += len(b)
    return out, total


def shrink_buffers(buffers_meta: list[dict]) -> list[bytes]:
    images = load_images(buffers_meta)  # декодируем один раз

    for tier in TIERS:
        out, total = compress_all(images, tier)
        if total <= MAX_TOTAL_DECODED:
            return out

    for md in range(176, 79, -16):
        q = max(36, 44 - (176 - md) // 24)
        out, total = compress_all(images, {"max_dim": md, "q": q})
        if total <= MAX_TOTAL_DECODED:
            return out

    out, total = compress_all(images, {"max_dim": 64, "q": 32})
    if total <= MAX_TOTAL_DECODED:
        return out

    raise RuntimeError(
        "Не удалось уложить изображения в лимит API (~30 МБ после base64). "
        "Уменьшите файлы или их число."
    )


# ─────────────────────────────────────────────────────────────
# Промпт (загружается из файла prompt.txt)
# ─────────────────────────────────────────────────────────────

PROMPT_FILE = ROOT / "prompt.txt"


def load_prompt() -> str:
    if not PROMPT_FILE.is_file():
        raise FileNotFoundError(
            f"Файл промпта не найден: {PROMPT_FILE}\n"
            "Убедитесь, что prompt.txt лежит рядом со скриптом."
        )
    return PROMPT_FILE.read_text(encoding="utf-8").strip()


# ─────────────────────────────────────────────────────────────
# SSL
# ─────────────────────────────────────────────────────────────

def requests_verify_bundle() -> bool | str:
    flag = os.environ.get("OPENROUTER_INSECURE_SSL", "").strip().lower()
    if flag in ("1", "true", "yes", "on"):
        print("ВНИМАНИЕ: проверка TLS отключена (OPENROUTER_INSECURE_SSL).", file=sys.stderr)
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
        truststore.inject_into_ssl()
        return True
    except ImportError:
        return certifi.where()


# ─────────────────────────────────────────────────────────────
# API
# ─────────────────────────────────────────────────────────────

def extract_image_from_response(data: dict) -> tuple[bytes, str]:
    msg = (data.get("choices") or [{}])[0].get("message") or {}
    images = msg.get("images") or []
    if not images:
        raise ValueError(
            "В ответе API нет поля images: " + json.dumps(data, ensure_ascii=False)[:800]
        )
    url = (images[0].get("image_url") or {}).get("url") or ""
    m = re.match(r"^data:([^;]+);base64,(.+)$", url, re.DOTALL)
    if not m:
        raise ValueError("Неожиданный формат картинки в ответе.")
    mime = m.group(1)
    raw_b64 = m.group(2)
    return base64.b64decode(raw_b64, validate=True), mime


def call_api(
    content: list[dict],
    headers: dict,
    verify: bool | str,
    step_label: str,
) -> dict:
    """Выполняет запрос к OpenRouter с retry-логикой. Возвращает распаршенный JSON."""
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": content}],
        "modalities": ["image", "text"],
        #"image_config": {"aspect_ratio": "1:4"},   # вертикальный баннер
    }

    payload_bytes = len(json.dumps(payload).encode("utf-8"))
    img_count = sum(1 for c in content if c.get("type") == "image_url")
    print(
        f"[{step_label}] Изображений в запросе: {img_count}, "
        f"размер JSON: {payload_bytes // 1024} КБ",
        file=sys.stderr,
    )

    data: dict = {}
    for attempt in range(1, MAX_RETRIES + 1):
        print(f"[{step_label}] Запрос к OpenRouter… (попытка {attempt}/{MAX_RETRIES})", file=sys.stderr)
        try:
            r = requests.post(
                OPENROUTER_URL,
                headers=headers,
                json=payload,
                timeout=300,
                verify=verify,
            )
        except requests.exceptions.SSLError as e:
            print(
                "Ошибка TLS при обращении к openrouter.ai.\n"
                "  • pip install -r requirements.txt (нужны truststore, certifi)\n"
                "  • Либо: SSL_CERT_FILE=путь\\к\\corp.pem\n"
                "  • Либо временно: OPENROUTER_INSECURE_SSL=1\n"
                f"Исходное сообщение: {e}",
                file=sys.stderr,
            )
            raise
        except requests.exceptions.ConnectionError as e:
            print(f"[{step_label}] Ошибка соединения (попытка {attempt}): {e}", file=sys.stderr)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
                continue
            raise
        except requests.exceptions.Timeout:
            print(f"[{step_label}] Таймаут запроса (попытка {attempt}).", file=sys.stderr)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
                continue
            raise RuntimeError(f"[{step_label}] Все попытки исчерпаны. Сервер не отвечает.")

        try:
            data = r.json()
        except json.JSONDecodeError:
            print(f"[{step_label}] Ответ не JSON:", r.text[:500], file=sys.stderr)
            raise RuntimeError(f"[{step_label}] Некорректный JSON в ответе.")

        api_err = data.get("error")
        if api_err or not r.ok:
            err_info = api_err if isinstance(api_err, dict) else {}
            err_code = err_info.get("code") or r.status_code
            err_msg  = err_info.get("message", r.reason or "неизвестно")

            if err_code == 524:
                print(
                    f"[{step_label}] Таймаут провайдера 524 (попытка {attempt}).",
                    file=sys.stderr,
                )
                if attempt < MAX_RETRIES:
                    print(f"Повтор через {RETRY_DELAY} сек…", file=sys.stderr)
                    time.sleep(RETRY_DELAY)
                    continue
                raise RuntimeError(
                    f"[{step_label}] Все попытки исчерпаны (524).\n"
                    "  • Уменьшите число изображений (3–5 вместо 10).\n"
                    "  • Попробуйте позже — возможна перегрузка провайдера."
                )

            if err_code == 400:
                is_generic = err_msg.strip().lower() in (
                    "provider returned error", "provider returned error.", ""
                )
                print(
                    f"[{step_label}] Ошибка 400 (попытка {attempt}): {err_msg}\n"
                    f"Полный ответ: {json.dumps(data, ensure_ascii=False)[:800]}",
                    file=sys.stderr,
                )
                if is_generic and attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
                    continue
                raise RuntimeError(f"[{step_label}] Ошибка 400: {err_msg}")

            raise RuntimeError(
                f"[{step_label}] Ошибка OpenRouter {err_code}: {err_msg}\n"
                f"Полный ответ: {json.dumps(data, ensure_ascii=False)[:800]}"
            )

        break  # успех

    return data


# ─────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────

def main() -> int:
    load_dotenv(ROOT / ".env")

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        print("Задайте OPENROUTER_API_KEY в файле .env или переменной окружения.", file=sys.stderr)
        return 1

    if not INPUT_DIR.is_dir():
        INPUT_DIR.mkdir(parents=True, exist_ok=True)
        print(
            "Создана папка «входные». Положите туда 3–10 изображений и запустите снова.",
            file=sys.stderr,
        )
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    names = sorted(
        [p.name for p in INPUT_DIR.iterdir() if p.is_file() and mime_from_name(p.name)],
        key=lambda x: x.lower(),
    )

    if not (3 <= len(names) <= 10):
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
        if not mime:
            print(f"Неподдерживаемый формат: {name}", file=sys.stderr)
            return 1
        buffers_meta.append({"buffer": buf, "mime": mime, "name": name})

    print("Сжатие изображений под лимит API…", file=sys.stderr)
    try:
        final_buffers = shrink_buffers(buffers_meta)
    except Exception as e:
        print(str(e), file=sys.stderr)
        return 1

    verify = requests_verify_bundle()

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/rich-content-cli",
        "X-Title": "Rich Content Python",
    }

    # ── Единый запрос: анализ цвета + OCR + компоновка ──────────────────────
    print("\nЗапрос к OpenRouter…", file=sys.stderr)

    try:
        prompt_text = load_prompt()
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 1

    content: list[dict] = [{"type": "text", "text": prompt_text}]
    for buf in final_buffers:
        b64 = base64.b64encode(buf).decode("ascii")
        content.append(
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        )

    try:
        data = call_api(content, headers, verify, "Генерация")
    except Exception as e:
        print(str(e), file=sys.stderr)
        return 1

    try:
        out_bytes, out_mime = extract_image_from_response(data)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1

    ext = "png" if "png" in out_mime.lower() else "jpg"
    out_path = OUTPUT_DIR / f"rich-{int(time.time() * 1000)}.{ext}"
    out_path.write_bytes(out_bytes)

    try:
        with Image.open(out_path) as im:
            w, h = im.size
        print(f"Размер результата: {w}×{h} px", file=sys.stderr)
        if w % 800 != 0 or h % 500 != 0:
            print(
                f"ВНИМАНИЕ: размер {w}×{h} не кратен 800×500 px.",
                file=sys.stderr,
            )
    except Exception as e:
        print(f"Не удалось проверить размер результата: {e}", file=sys.stderr)

    print(out_path.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())