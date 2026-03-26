#!/usr/bin/env python3
"""
traductor.py — Traductor de PDF usando pymupdf4llm + Ollama/Aphrodite + Pandoc

Pipeline:
  PDF → pymupdf4llm → Markdown por páginas → chunks con contexto
      → LLM (Ollama o Aphrodite) → Markdown traducido → Pandoc → PDF final

Uso:
  conda run -n pdf-translator python traductor.py libro.pdf
  conda run -n pdf-translator python traductor.py libro.pdf --model gemma3:12b --paginas 1-50
  conda run -n pdf-translator python traductor.py libro.pdf --backend aphrodite --api-base http://localhost:2242/v1
"""

import os
import sys
import json
import argparse
import subprocess
import textwrap
import time
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

import requests
import pymupdf4llm

# ─────────────────────────────────────────────────────────────
# CONFIGURACIÓN POR DEFECTO (se sobreescribe con config.yaml y CLI)
# ─────────────────────────────────────────────────────────────
DEFAULTS = {
    "model":        "translategemma:12b",
    "src":          "English",
    "dst":          "Español",
    "backend":      "vllm",              # "ollama", "vllm" o "aphrodite"
    "api_base":     "http://localhost:8000/v1",
    "chunk_tokens": 900,                 # aprox tokens por chunk de traducción
    "timeout":      120,                 # segundos por petición al LLM
    "retries":      3,                   # reintentos ante fallo de red
    "paginas":      None,
    "output":       "output",
    "no_pdf":       False,
    "limpio":       False,
}

# Intentar cargar config.yaml global
config_path = Path("config.yaml")
if config_path.exists():
    if yaml is None:
        print("[!] config.yaml encontrado pero 'PyYAML' no está instalado. Ejecuta: pip install PyYAML")
    else:
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                _yml_cfg = yaml.safe_load(f) or {}
                for _k, _v in _yml_cfg.items():
                    # Homologar variantes o guiones medios
                    _k_norm = _k.replace("-", "_")
                    if _k_norm == "src_lang": _k_norm = "src"
                    if _k_norm == "dst_lang": _k_norm = "dst"
                    DEFAULTS[_k_norm] = _v
        except Exception as _e:
            print(f"[-] Advertencia: No se pudo leer config.yaml ({_e})")

# ─────────────────────────────────────────────────────────────
# PANDOC: plantilla LaTeX mínima para PDF elegante
# ─────────────────────────────────────────────────────────────
PANDOC_CMD_BASE = [
    "pandoc",
    "-f", "markdown-tex_math_dollars-raw_tex",
    "--pdf-engine=xelatex",
    "-V", "geometry:margin=2.5cm",
    "-V", "fontsize=11pt",
    "-V", "mainfont=DejaVu Serif",
    "-V", "sansfont=DejaVu Sans",
    "-V", "monofont=DejaVu Sans Mono",
    "-V", "linestretch=1.35",
    "-V", "colorlinks=true",
    "-V", "linkcolor=black",
    "-V", "lang=es",
    "--standalone",
]

# ─────────────────────────────────────────────────────────────
# UTILIDADES
# ─────────────────────────────────────────────────────────────

def log(msg: str, level: str = "INFO"):
    prefix = {"INFO": "[*]", "OK": "[+]", "WARN": "[-]", "ERR": "[!]"}.get(level, "[?]")
    print(f"{prefix} {msg}", flush=True)


def estimar_tokens(texto: str) -> int:
    """Estimación rápida: ~1 token cada 4 caracteres."""
    return len(texto) // 4


def get_iso_code(lang: str) -> str:
    """Mapea nombres comunes de idiomas a códigos ISO 2-letras."""
    lang_map = {
        "english": "en", "inglés": "en",
        "spanish": "es", "español": "es", "castellano": "es",
        "french": "fr", "francés": "fr",
        "german": "de", "alemán": "de",
        "portuguese": "pt", "portugués": "pt",
        "italian": "it", "italiano": "it",
        "chinese": "zh", "chino": "zh",
        "japanese": "ja", "japonés": "ja",
        "russian": "ru", "ruso": "ru",
    }
    return lang_map.get(lang.lower(), lang.lower()[:2])


def rango_paginas(spec: str, total: int) -> list[int]:
    """Convierte '1-10,15,20-25' en lista de índices 0-based."""
    indices = set()
    for parte in spec.split(","):
        parte = parte.strip()
        if "-" in parte:
            a, b = parte.split("-", 1)
            indices.update(range(int(a) - 1, min(int(b), total)))
        else:
            n = int(parte) - 1
            if 0 <= n < total:
                indices.add(n)
    return sorted(indices)


# ─────────────────────────────────────────────────────────────
# VERIFICACIÓN DE SERVICIOS
# ─────────────────────────────────────────────────────────────

def verificar_ollama(api_base: str, model: str):
    log("Verificando Ollama...")
    try:
        r = requests.get(api_base.replace("/api", ""), timeout=5)
        if r.status_code != 200:
            raise ConnectionError()
    except Exception:
        log("Ollama no responde en " + api_base, "ERR")
        log("Asegúrate de ejecutar: ollama serve", "ERR")
        sys.exit(1)

    # Precargar modelo
    log(f"Precargando modelo {model}...")
    try:
        requests.post(
            f"{api_base}/generate",
            json={"model": model, "prompt": "", "keep_alive": "10m"},
            timeout=3,
        )
    except requests.exceptions.Timeout:
        pass  # Normal al cargar el modelo por primera vez
    except Exception as e:
        log(f"Advertencia al precargar: {e}", "WARN")
    log("Ollama listo.", "OK")


def verificar_openai_compatible(api_base: str, model: str, backend_name: str = "vLLM/Aphrodite"):
    log(f"Verificando {backend_name} (OpenAI-compatible)...")
    # Limpiar api_base: asegurar que no termine en /chat/completions o /completions
    base_url = api_base.rstrip("/")
    if base_url.endswith("/v1"):
        models_url = f"{base_url}/models"
    else:
        models_url = f"{base_url}/v1/models" if "/v1" not in base_url else f"{base_url}/models"

    try:
        r = requests.get(models_url, timeout=10)
        r.raise_for_status()
        modelos = [m["id"] for m in r.json().get("data", [])]
        log(f"Modelos disponibles en {backend_name}: {modelos}")
        if model not in modelos:
             log(f"ERROR: El modelo '{model}' no está cargado en {backend_name}.", "ERR")
             log(f"Modelos permitidos: {modelos}", "ERR")
             log("Por favor, actualiza el campo 'model' en config.yaml", "ERR")
             sys.exit(1)
    except Exception as e:
        log(f"No se pudo conectar con {backend_name} en {models_url}: {e}", "ERR")
        log("Asegúrate de que el servidor está corriendo y la URL es correcta.", "ERR")
        sys.exit(1)
    log(f"{backend_name} listo.", "OK")


# ─────────────────────────────────────────────────────────────
# EXTRACCIÓN DE TEXTO (pymupdf4llm → Markdown por página)
# ─────────────────────────────────────────────────────────────

def extraer_markdown_por_pagina(pdf_path: str, paginas: list[int], images_dir: Path) -> list[dict]:
    """
    Usa pymupdf4llm para extraer cada página como Markdown limpio.
    Las imágenes se guardan en images_dir y se referencian en el Markdown.
    Devuelve lista de {"page": N, "markdown": "..."}
    """
    log(f"Extrayendo texto e imágenes con pymupdf4llm ({len(paginas)} páginas)...")
    images_dir_abs = images_dir.resolve()
    images_dir_abs.mkdir(parents=True, exist_ok=True)

    # page_chunks=True devuelve una lista de dicts con .text y .metadata
    chunks = pymupdf4llm.to_markdown(
        str(Path(pdf_path).resolve()),
        pages=paginas,
        page_chunks=True,
        show_progress=False,
        write_images=True,
        image_path=str(images_dir_abs),
        image_format="png",
    )

    resultado = []
    for i, chunk in enumerate(chunks):
        # pymupdf4llm ≥0.0.8 usa la clave 'text'; versiones anteriores usan 'markdown'
        texto = chunk.get("text") or chunk.get("markdown", "")
        # 'page' puede no estar en metadata cuando se usa el filtro pages=
        num_pag = chunk.get("metadata", {}).get("page") or paginas[i] + 1

        texto = texto.strip()
        # Ignorar páginas que sólo tienen marcadores de imagen y espacios
        texto_util = texto.replace("==> picture", "").replace("intentionally omitted", "").strip()
        if not texto_util or all(c in "<>=*#- \n" for c in texto_util):
            log(f"  Página {num_pag}: sólo imagen, omitida.")
            continue

        resultado.append({"page": num_pag, "markdown": texto})

    n_imgs = len(list(images_dir.glob("*.png")))
    log(f"Extracción completada: {len(resultado)} páginas con texto, {n_imgs} imágenes guardadas.", "OK")
    return resultado


# ─────────────────────────────────────────────────────────────
# AGRUPACIÓN EN CHUNKS CON VENTANA DE CONTEXTO
# ─────────────────────────────────────────────────────────────

import re

def agrupar_en_chunks(paginas_md: list[dict], max_tokens: int) -> list[dict]:
    """
    Agrupa el texto en lotes semánticos (por párrafo) respetando max_tokens.
    Reconstruye inteligentemente oraciones rotas por saltos de página.
    Cada chunk incluye metadatos de las páginas que abarca.
    """
    bloques = []
    
    # 1. Descomponer todo en párrafos individuales, uniendo frases partidas entre páginas
    for pmd in paginas_md:
        texto = pmd["markdown"].strip()
        if not texto:
            continue
            
        parrafos = [p.strip() for p in texto.split("\n\n") if p.strip()]
        if not parrafos:
            continue
            
        # Si hay un bloque anterior y termina sin puntuación final, asumimos que un salto de página cortó la frase
        if bloques and bloques[-1]["texto"] and bloques[-1]["texto"][-1] not in ".!?\":;)]>":
            # Prevenir la fusión si el siguiente párrafo es un elemento de bloque de Markdown (título, lista, imagen)
            is_block = re.match(r"^(#{1,6}\s|\* |\- |\d+\. |!\[|> )", parrafos[0])
            if not is_block:
                bloques[-1]["texto"] += " " + parrafos[0]
                bloques[-1]["paginas"].add(pmd["page"])
                parrafos = parrafos[1:] # Procesar el resto normalmente
            
        for p in parrafos:
            bloques.append({"texto": p, "paginas": {pmd["page"]}})

    # 2. Empaquetar los párrafos en chunks de traducción según max_tokens
    chunks = []
    buffer_texto = ""
    buffer_paginas = set()

    for b in bloques:
        tokens_nuevos = estimar_tokens(b["texto"])
        tokens_actuales = estimar_tokens(buffer_texto)
        
        if tokens_actuales + tokens_nuevos > max_tokens and buffer_texto:
            chunks.append({"paginas": sorted(list(buffer_paginas)), "texto": buffer_texto.strip()})
            buffer_texto = b["texto"]
            buffer_paginas = set(b["paginas"])
        else:
            buffer_texto += "\n\n" + b["texto"] if buffer_texto else b["texto"]
            buffer_paginas.update(b["paginas"])

    if buffer_texto.strip():
        chunks.append({"paginas": sorted(list(buffer_paginas)), "texto": buffer_texto.strip()})

    log(f"Texto extraído re-ensamblado en {len(chunks)} chunks semánticos (por párrafo).", "OK")
    return chunks


# ─────────────────────────────────────────────────────────────
# TRADUCCIÓN
# ─────────────────────────────────────────────────────────────

# Mensaje de sistema (instruye al modelo sobre su rol)
SYSTEM_PROMPT = (
    "You are a professional literary translator. "
    "Your sole task is to translate the text the user sends you within <text> tags. "
    "Output ONLY the translated text — absolutely no explanations, "
    "notes, preamble, or repetition of these instructions. "
    "Preserve ALL Markdown formatting (headers #, bold **, italic *, lists, images, etc.). "
    "Do NOT translate: URLs, emails, image paths, code blocks, or untranslatable proper nouns. "
    "Do NOT add ellipsis (...) unless the original has them. "
    "Maintain all paragraph breaks and document structure exactly. "
    "Ignore entirely the <context> tags, they are only for reference."
)


def construir_user_message(texto: str, src: str, dst: str, contexto: str, model: str = "") -> str:
    """
    Mensaje del usuario: detecta si es TranslateGemma para usar el formato de marcas estricto.
    """
    is_translategemma = "translategemma" in model.lower()

    if is_translategemma:
        # Formato oficial: <<<source>>>{src}<<<target>>>{dst}<<<text>>>{text}
        src_iso = get_iso_code(src)
        dst_iso = get_iso_code(dst)
        
        # El contexto se inyecta al inicio del texto si existe
        full_text = f"<context>\n{contexto}\n</context>\n\n{texto}" if contexto else texto
        return f"<<<source>>>{src_iso}<<<target>>>{dst_iso}<<<text>>>{full_text}"

    # Formato estándar para otros modelos
    partes = [f"Translate the text within <text> from {src} to {dst}."]
    if contexto:
        partes.append(f"<context>\n{contexto}\n</context>")
    partes.append(f"<text>\n{texto}\n</text>")
    return "\n".join(partes)


IMPORTANT_PREFIXES = [
    "translated text:", "translation:", "here is the translation:",
    "here's the translation:", "the translation is:",
    "texto traducido:", "traducción:", "aquí está la traducción:",
    "rules:", "reglas:", "context:", "contexto:",
    "text to translate:", "texto a traducir:",
    "=== Context from previous section", "=== Contexto de la sección",
    "[Context from previous", "[Contexto de la sección",
]


import re

def limpiar_traduccion(texto: str) -> str:
    """
    Elimina artefactos del prompt aislando el contenido a través de parseo XML rudimentario
    o limpieza línea a línea por si el modelo falla.
    """
    # 1. Si el modelo escupió el contexto rodeado de <context>...</context>, eliminarlo por completo
    texto = re.sub(r"<context>.*?</context>", "", texto, flags=re.DOTALL | re.IGNORECASE)
    texto = texto.strip()

    # 2. Si el modelo envolvió su respuesta en <text>...</text>, extraer solo eso
    match = re.search(r"<text>\s*(.*?)\s*</text>", texto, flags=re.DOTALL | re.IGNORECASE)
    if match:
        texto = match.group(1).strip()
    
    # 3. Remover etiquetas huérfanas
    texto = re.sub(r"</?text>", "", texto, flags=re.IGNORECASE)
    texto = re.sub(r"</?context>", "", texto, flags=re.IGNORECASE)

    # 4. Strip simple prefix on single-paragraph response
    linea_lower = texto.lower().strip()
    for prefijo in IMPORTANT_PREFIXES:
        if linea_lower.startswith(prefijo):
            texto = texto[len(prefijo):].strip()
            linea_lower = texto.lower().strip()

    # 5. Filtrar líneas que parezcan artefactos del prompt (fallbacks adicionales)
    LINEAS_PROMPT = {
        "rules:", "reglas:", "context:", "contexto:",
        "text to translate:", "texto a traducir:",
        "translate from", "traducir de",
        "=== context", "=== contexto",
        "[context", "[contexto"
    }
    lineas_out = []
    skip_block = False
    for linea in texto.splitlines():
        linea_l = linea.lower().strip()
        # Detectar inicio de bloque de prompt
        if any(linea_l.startswith(m) for m in LINEAS_PROMPT):
            skip_block = True
            continue
        # Fin de bloque: línea en blanco despues de skip resetea
        if skip_block and linea_l == "":
            skip_block = False
            continue
        if skip_block:
            continue
        lineas_out.append(linea)

    return "\n".join(lineas_out).strip()


def traducir_ollama(texto: str, src: str, dst: str, contexto: str,
                    api_base: str, model: str, timeout: int, retries: int) -> str:
    """
    Usa /api/chat con roles system/user para evitar que el modelo
    repita el prompt en su respuesta (prompt bleeding).
    """
    user_msg = construir_user_message(texto, src, dst, contexto)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        "stream": False,
        "options": {"temperature": 0.3},
    }

    for intento in range(1, retries + 1):
        try:
            r = requests.post(f"{api_base}/chat", json=payload, timeout=timeout)
            r.raise_for_status()
            traduccion = r.json().get("message", {}).get("content", "").strip()
            return limpiar_traduccion(traduccion)
        except Exception as e:
            log(f"Intento {intento}/{retries} fallido: {e}", "WARN")
            if intento < retries:
                time.sleep(3 * intento)

    log("Todos los intentos fallaron. Devolviendo texto original.", "ERR")
    return texto  # fallback: preservar original


def traducir_aphrodite(texto: str, src: str, dst: str, contexto: str,
                        api_base: str, model: str, timeout: int, retries: int) -> str:
    # Intentar detectar si el modelo requiere el endpoint legado /completions
    # (vLLM suele preferir /v1/chat/completions para modelos -it)
    # Si el usuario quiere forzar completions, puede incluirlo en el modelo o backend, 
    # pero aquí intentaremos ser inteligentes.
    
    # FORZAR chat/completions por defecto para vLLM/Aphrodite models modernos
    endpoint = "/chat/completions"
    
    # Formato estándar para el 99% de los modelos (Chat API)
    user_msg = construir_user_message(texto, src, dst, contexto, model=model)
    
    # Si es TranslateGemma, NO incluimos SYSTEM_PROMPT porque el modelo es zero-shot 
    # y vLLM puede rechazar mensajes que no empiecen con <<<source>>> si el template es estricto.
    is_translategemma = "translategemma" in model.lower()
    
    if is_translategemma:
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": user_msg},
            ],
            "temperature": 0.3,
            "max_tokens": 4096,
        }
    else:
        combined_msg = f"{SYSTEM_PROMPT}\n\n{user_msg}"
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": combined_msg},
            ],
            "temperature": 0.3,
            "max_tokens": 4096,
        }

    for intento in range(1, retries + 1):
        try:
            r = requests.post(f"{api_base}{endpoint}", json=payload, timeout=timeout)
            r.raise_for_status()
            
            resp_json = r.json()
            if "choices" in resp_json and len(resp_json["choices"]) > 0:
                choice = resp_json["choices"][0]
                if "message" in choice:
                    traduccion = choice["message"]["content"].strip()
                elif "text" in choice:
                    traduccion = choice["text"].strip()
                else:
                    traduccion = ""
            else:
                traduccion = ""
                
            return limpiar_traduccion(traduccion)
        except Exception as e:
            log(f"Intento {intento}/{retries} fallido: {e}", "WARN")
            if intento < retries:
                time.sleep(3 * intento)

    log("Todos los intentos fallaron. Devolviendo texto original.", "ERR")
    return texto


def traducir_chunk(chunk_texto: str, src: str, dst: str, contexto: str, cfg: dict) -> str:
    if cfg["backend"] in ["aphrodite", "vllm"]:
        return traducir_aphrodite(
            chunk_texto, src, dst, contexto,
            cfg["api_base"], cfg["model"], cfg["timeout"], cfg["retries"],
        )
    return traducir_ollama(
        chunk_texto, src, dst, contexto,
        cfg["api_base"], cfg["model"], cfg["timeout"], cfg["retries"],
    )


# ─────────────────────────────────────────────────────────────
# GENERACIÓN DE PDF VÍA PANDOC
# ─────────────────────────────────────────────────────────────

def generar_pdf(markdown_path: Path, pdf_path: Path, dst_lang: str, images_dir: Path = None):
    log(f"Generando PDF final con Pandoc: {pdf_path}")

    # Determinar el código ISO para el idioma destino
    lang_map = {
        "español": "es", "spanish": "es",
        "french": "fr", "français": "fr",
        "german": "de", "deutsch": "de",
        "portuguese": "pt", "português": "pt",
        "italian": "it", "italiano": "it",
        "english": "en",
    }
    lang_code = lang_map.get(dst_lang.lower(), "es")

    # pandoc necesita saber dónde buscar las imágenes referenciadas en el Markdown
    resource_path = str(markdown_path.parent)
    if images_dir and images_dir.exists():
        resource_path = f"{resource_path}:{images_dir}"

    cmd = PANDOC_CMD_BASE + [
        "-V", f"lang={lang_code}",
        f"--resource-path={resource_path}",
        "-o", str(pdf_path),
        str(markdown_path),
    ]

    resultado = subprocess.run(cmd, capture_output=True, text=True)
    if resultado.returncode != 0:
        log("Pandoc reportó un problema:", "WARN")
        log(resultado.stderr[:500], "WARN")
        # Intentar sin xelatex (fallback a pdflatex)
        cmd_fb = [c if c != "xelatex" else "pdflatex" for c in cmd]
        cmd_fb = [c for c in cmd_fb if not c.startswith("DejaVu")]
        resultado2 = subprocess.run(cmd_fb, capture_output=True, text=True)
        if resultado2.returncode != 0:
            log("Pandoc falló en el fallback. Revisa la instalación de LaTeX.", "ERR")
            log(resultado2.stderr[:300], "ERR")
            return False

    log(f"PDF guardado: {pdf_path}", "OK")
    return True


# ─────────────────────────────────────────────────────────────
# CACHÉ DE PROGRESO (resume)
# ─────────────────────────────────────────────────────────────

def cargar_cache(cache_path: Path) -> dict:
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            log(f"Caché encontrada: {len(data.get('chunks_done', []))} chunks ya traducidos.")
            return data
        except Exception:
            pass
    return {"chunks_done": [], "translated_parts": []}


def guardar_cache(cache_path: Path, data: dict):
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Traductor de PDF usando pymupdf4llm + LLM + Pandoc",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
          Ejemplos:
            python traductor.py libro.pdf
            python traductor.py libro.pdf --model llama3:8b --dst Español
            python traductor.py libro.pdf --paginas 1-30,50-60
            python traductor.py libro.pdf --backend aphrodite --api-base http://localhost:2242/v1
        """),
    )

    parser.add_argument("archivo",          help="Ruta al archivo PDF")
    parser.add_argument("--model",          default=DEFAULTS["model"],        help="Modelo LLM")
    parser.add_argument("--src",            default=DEFAULTS["src"],          help="Idioma origen")
    parser.add_argument("--dst",            default=DEFAULTS["dst"],          help="Idioma destino")
    parser.add_argument("--backend",        default=DEFAULTS["backend"],      choices=["ollama", "aphrodite", "vllm"])
    parser.add_argument("--api-base",       default=DEFAULTS["api_base"],     help="URL base de la API del LLM")
    parser.add_argument("--chunk-tokens",   default=DEFAULTS["chunk_tokens"], type=int)
    parser.add_argument("--timeout",        default=DEFAULTS["timeout"],      type=int)
    parser.add_argument("--retries",        default=DEFAULTS["retries"],      type=int)
    parser.add_argument("--paginas",        default=DEFAULTS["paginas"],      help="Rango de páginas ej: 1-50,80-100")
    parser.add_argument("--output",         default=DEFAULTS["output"],       help="Carpeta de salida")
    parser.add_argument("--no-pdf",         action="store_true", default=DEFAULTS["no_pdf"], help="Sólo generar .md, no PDF")
    parser.add_argument("--limpio",         action="store_true", default=DEFAULTS["limpio"], help="Ignorar caché y empezar de cero")

    args = parser.parse_args()

    # Verificar que el archivo existe
    pdf_path = Path(args.archivo).resolve()
    if not pdf_path.exists():
        log(f"Archivo no encontrado: {pdf_path}", "ERR")
        sys.exit(1)

    # Configuración del LLM
    cfg = {
        "model":    args.model,
        "backend":  args.backend,
        "api_base": args.api_base,
        "timeout":  args.timeout,
        "retries":  args.retries,
    }

    # Directorios de salida
    # IMPORTANTE: pymupdf4llm reemplaza espacios por _ en los nombres de imagen.
    # su función md_path() relativiza el path desde cwd e incluye el stem sanitizado.
    # Para que el directorio exista cuando MuPDF intente escribir, usamos el stem
    # sanitizado tanto en out_dir como en images_dir.
    stem_orig      = pdf_path.stem                         # nombre original (puede tener espacios)
    stem_safe      = stem_orig.replace(" ", "_")           # sin espacios
    out_dir        = Path(args.output) / stem_safe
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir     = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    cache_path     = out_dir / f"{stem_safe}_progreso.json"
    md_out         = out_dir / f"{stem_safe}_traducido.md"
    pdf_out        = out_dir / f"{stem_safe}_traducido.pdf"


    print()
    print("=" * 60)
    print(f"  PDF Translator — {args.src} → {args.dst}")
    print(f"  Backend : {args.backend} | Modelo: {args.model}")
    print(f"  Entrada : {pdf_path.name}")
    print(f"  Salida  : {out_dir}/")
    print("=" * 60)
    print()

    # ── Verificar servicio LLM ──────────────────────────────
    if args.backend == "ollama":
        verificar_ollama(args.api_base, args.model)
    else:
        # Normalizar api_base para OpenAI-compatible (vLLM/Aphrodite)
        # Si termina en /chat/completions o /completions, quitarlo
        base = args.api_base.rstrip("/")
        for suffix in ["/chat/completions", "/completions"]:
            if base.endswith(suffix):
                base = base[:-len(suffix)]
        cfg["api_base"] = base.rstrip("/")
        
        verificar_openai_compatible(cfg["api_base"], args.model, args.backend.upper())

    # ── Determinar páginas a procesar ──────────────────────
    import fitz
    doc = fitz.open(str(pdf_path))
    total_paginas = len(doc)
    doc.close()

    if args.paginas:
        paginas_idx = rango_paginas(args.paginas, total_paginas)
    else:
        paginas_idx = list(range(total_paginas))

    log(f"Total páginas en el PDF: {total_paginas}  |  A procesar: {len(paginas_idx)}")

    # ── Extracción: PDF → Markdown por página ─────────────
    paginas_md = extraer_markdown_por_pagina(str(pdf_path), paginas_idx, images_dir)

    # ── Agrupar en chunks ──────────────────────────────────
    chunks = agrupar_en_chunks(paginas_md, args.chunk_tokens)

    # ── Cargar caché ───────────────────────────────────────
    cache = {"chunks_done": [], "translated_parts": []} if args.limpio else cargar_cache(cache_path)

    # ── Traducción chunk a chunk ───────────────────────────
    print()
    log(f"Iniciando traducción: {len(chunks)} chunks...")
    contexto_previo = ""

    for i, chunk in enumerate(chunks):
        chunk_id = f"chunk_{i:04d}"
        paginas_str = ",".join(str(p) for p in chunk["paginas"])

        if chunk_id in cache["chunks_done"]:
            log(f"  [{i+1}/{len(chunks)}] páginas {paginas_str} — ya traducido (skip)")
            # Recuperar el texto traducido para mantener el contexto
            idx = cache["chunks_done"].index(chunk_id)
            contexto_previo = cache["translated_parts"][idx][-300:]
            continue

        log(f"  [{i+1}/{len(chunks)}] Traduciendo páginas {paginas_str} "
            f"(~{estimar_tokens(chunk['texto'])} tokens)...")

        traduccion = traducir_chunk(
            chunk["texto"],
            args.src,
            args.dst,
            contexto_previo,
            cfg,
        )

        # Actualizar contexto con el final del chunk traducido (últimos 300 chars)
        contexto_previo = traduccion[-300:] if traduccion else ""

        # Guardar en caché
        cache["chunks_done"].append(chunk_id)
        cache["translated_parts"].append(traduccion)
        guardar_cache(cache_path, cache)

    # ── Ensamblar Markdown final ───────────────────────────
    log("Ensamblando Markdown traducido...")
    
    # Unimos los chunks con saltos de párrafo en lugar de líneas divisorias ---
    # Si un párrafo quedó cortado a la mitad por el tamaño del chunk, será más fluido.
    md_completo = "\n\n".join(cache["translated_parts"])
    
    import re
    # Asegurar que los encabezados Markdown tengan una línea en blanco antes y después.
    # Romper los que se hayan colado en la misma línea por errores del LLM
    md_completo = re.sub(r"([^\n])\s+(#{1,6}\s+\*\*?[A-ZÍÓÚÁÉa-z0-9])", r"\1\n\n\2", md_completo)
    # Rellenar con línea en blanco los que solo tienen 1 salto de línea
    md_completo = re.sub(r"([^\n])\n(#{1,6}\s)", r"\1\n\n\2", md_completo)
    
    md_out.write_text(md_completo, encoding="utf-8")
    log(f"Markdown guardado: {md_out}", "OK")

    # ── Generar PDF ────────────────────────────────────────
    if not args.no_pdf:
        exito = generar_pdf(md_out, pdf_out, args.dst, images_dir)
        if exito:
            log(f"\n¡ÉXITO! PDF traducido: {pdf_out}", "OK")
        else:
            log(f"PDF no generado. El Markdown traducido está en: {md_out}", "WARN")
    else:
        log(f"Modo --no-pdf: sólo Markdown. Archivo: {md_out}", "OK")

    # ── Limpiar caché si todo fue bien ─────────────────────
    if not args.no_pdf and pdf_out.exists():
        try:
            cache_path.unlink()
        except Exception:
            pass

    print()
    print("=" * 60)
    print("  PROCESO TERMINADO")
    print("=" * 60)


if __name__ == "__main__":
    main()
