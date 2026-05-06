let sharpMod = null;
try {
  sharpMod = require('sharp');
} catch (e) {
  sharpMod = null;
}

const all = $input.all();
const parts = [];
for (let itemIndex = 0; itemIndex < all.length; itemIndex++) {
  const item = all[itemIndex];
  if (!item.binary) continue;
  for (const key of Object.keys(item.binary)) {
    parts.push({ itemIndex, key, bin: item.binary[key] });
  }
}

const MAX_BYTES = 5 * 1024 * 1024;
const ALLOWED = [
  'image/jpeg',
  'image/jpg',
  'image/png',
  'image/jfif',
  'image/pjpeg',
];
// Провайдер: «Downloaded image content cannot exceed 30MB» — лимит по сумме входных картинок;
// в теле запроса они ещё и в base64 (~+33% к размеру).
const MAX_TOTAL_DECODED = 16 * 1024 * 1024;

function fail(message, extra = {}) {
  return [{ json: { valid: false, error: message, ...extra } }];
}

if (parts.length < 3 || parts.length > 10) {
  return fail('Нужно от 3 до 10 изображений.', { count: parts.length });
}

const buffersMeta = [];

for (let i = 0; i < parts.length; i++) {
  const { itemIndex, key, bin } = parts[i];
  const mime = (bin.mimeType || '').toLowerCase();
  if (!ALLOWED.includes(mime)) {
    return fail('Допустимы только JPG/JPEG/JFIF и PNG.', { index: i, mime });
  }
  let buffer;
  try {
    buffer = await this.helpers.getBinaryDataBuffer(itemIndex, key);
  } catch (e) {
    return fail('Не удалось прочитать файл (binary).', {
      index: i,
      detail: String(e.message || e),
    });
  }
  if (buffer.length > MAX_BYTES) {
    return fail('Файл превышает 5 МБ.', { index: i, size: buffer.length });
  }
  buffersMeta.push({ buffer, mime });
}

async function compressOne(sharp, buf, mimeIn, maxDim, quality) {
  if (!sharp) return buf;
  const isPng = mimeIn.includes('png');
  let p = sharp(buf).rotate().resize(maxDim, maxDim, {
    fit: 'inside',
    withoutEnlargement: true,
  });
  if (isPng) {
    p = p.flatten({ background: { r: 255, g: 255, b: 255 } });
  }
  try {
    return await p.jpeg({ quality, mozjpeg: true }).toBuffer();
  } catch (e) {
    throw new Error('sharp: ' + String(e.message || e));
  }
}

const tiers = [
  { maxDim: 1024, q: 76 },
  { maxDim: 800, q: 72 },
  { maxDim: 640, q: 68 },
  { maxDim: 512, q: 64 },
  { maxDim: 400, q: 60 },
  { maxDim: 320, q: 56 },
  { maxDim: 256, q: 52 },
  { maxDim: 192, q: 48 },
];

async function compressAll(tier) {
  const out = [];
  let total = 0;
  for (const bm of buffersMeta) {
    const b = await compressOne(sharpMod, bm.buffer, bm.mime, tier.maxDim, tier.q);
    out.push(b);
    total += b.length;
  }
  return { out, total };
}

let finalBuffers = null;

if (sharpMod) {
  for (const tier of tiers) {
    try {
      const { out, total } = await compressAll(tier);
      if (total <= MAX_TOTAL_DECODED) {
        finalBuffers = out;
        break;
      }
    } catch (e) {
      return fail('Ошибка сжатия изображения.', { detail: String(e.message || e) });
    }
  }
  if (!finalBuffers) {
    for (let md = 176; md >= 80; md -= 16) {
      const q = Math.max(36, 44 - Math.floor((176 - md) / 24));
      try {
        const { out, total } = await compressAll({ maxDim: md, q });
        if (total <= MAX_TOTAL_DECODED) {
          finalBuffers = out;
          break;
        }
      } catch (e) {
        return fail('Ошибка сжатия изображения.', { detail: String(e.message || e) });
      }
    }
  }
  if (!finalBuffers) {
    try {
      const { out, total } = await compressAll({ maxDim: 64, q: 32 });
      if (total <= MAX_TOTAL_DECODED) {
        finalBuffers = out;
      }
    } catch (e) {
      return fail('Ошибка сжатия изображения.', { detail: String(e.message || e) });
    }
  }
} else {
  let total = 0;
  const out = [];
  for (const bm of buffersMeta) {
    total += bm.buffer.length;
    out.push(bm.buffer);
  }
  if (total <= MAX_TOTAL_DECODED) {
    finalBuffers = out;
  }
}

if (!finalBuffers) {
  const sumRaw = buffersMeta.reduce((s, b) => s + b.buffer.length, 0);
  if (!sharpMod) {
    const mib = (sumRaw / (1024 * 1024)).toFixed(2);
    const limitMb = (MAX_TOTAL_DECODED / (1024 * 1024)).toFixed(0);
    return fail(
      'Пакет sharp в n8n не найден — изображения не сжимаются. Без сжатия разрешена сумма файлов не больше ~' +
        limitMb +
        ' МиБ (сырые байты). У вас ~' +
        mib +
        ' МиБ (' +
        sumRaw +
        ' байт): это то же, что «24,5 МБ» в проводнике (разные округления и единицы МБ/МиБ). После кодирования base64 ~25 МБ станут >30 МБ для API. Установите sharp в окружение n8n (npm install sharp в каталоге установки или в Docker-образе) и перезапустите n8n.',
      { totalBytes: sumRaw, limitWithoutSharpMiB: Number(limitMb) },
    );
  }
  return fail(
    'Не удалось уложить изображения в лимит API даже при сильном сжатии. Уменьшите число файлов или разрешение исходников.',
    { totalBytesApprox: sumRaw },
  );
}

const PROMPT = [
  'Ты создаёшь один вертикальный РИЧ-баннер для маркетплейса: единое полотно визуального лендинга, не набор случайно склеенных картинок.',
  'Используй переданные изображения как референсы товара и стиля. Нумерация сверху вниз: изображение 1 … изображение N.',
  'Текст на баннере разрешён ТОЛЬКО из того, что реально читается на этих фото. Не придумывай состав, цифры, обещания и характеристики, если их нет на изображениях.',
  'Структура сверху вниз: Hero → краткие преимущества → при необходимости раскрытие → свойства/материалы (если видно на фото) → ассортимент только если на фото явно несколько SKU.',
  'Единая палитра и аккуратная типографика, читаемость.',
  'Сгенерируй одно итоговое изображение (вертикальный баннер).',
].join('\n');

const content = [{ type: 'text', text: PROMPT }];
const outMime = sharpMod ? 'image/jpeg' : null;

for (let i = 0; i < finalBuffers.length; i++) {
  const buf = finalBuffers[i];
  const mime = outMime || buffersMeta[i].mime;
  const b64 = buf.toString('base64');
  const dataUrl = 'data:' + mime + ';base64,' + b64;
  content.push({
    type: 'image_url',
    image_url: { url: dataUrl },
  });
}

const payload = {
  model: 'google/gemini-3.1-flash-image-preview',
  messages: [{ role: 'user', content }],
  modalities: ['image', 'text'],
  image_config: {
    aspect_ratio: '1:4',
    image_size: '2K',
  },
};

return [{ json: { valid: true, payload } }];
