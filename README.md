# 📚 PDF Translator — Traducción local con LLM

Traduce documentos PDF completos de forma local usando modelos de lenguaje (LLMs) ejecutados en tu propia máquina. Soporta **Ollama**, **vLLM** y **Aphrodite Engine** como backends.

**Pipeline:**
```
PDF → pymupdf4llm → Markdown por páginas → chunks semánticos
    → LLM (Ollama / vLLM / Aphrodite) → Markdown traducido → Pandoc → PDF
```

---

## ✨ Características

- **100% Local y Privado** — Sin APIs de pago ni datos enviados a terceros.
- **Multi-backend** — Ollama (CPU/GPU), vLLM (GPU, OpenAI-compatible), Aphrodite Engine.
- **Traducción Paralela** — Solicitudes concurrentes configurables (`workers`), aprovechando el batching de vLLM.
- **Reanudación automática** — Caché de progreso en JSON: si se interrumpe, continúa donde se quedó.
- **Contexto semántico** — En modo secuencial, el traductor recuerda los últimos párrafos para mantener coherencia.
- **Preserva formato Markdown** — Títulos, negritas, cursivas, listas e imágenes se mantienen intactos.
- **Generación de PDF** — Pandoc + XeLaTeX convierte el Markdown traducido en un PDF tipográfico final.
- **Configuración centralizada** — Todos los parámetros en `config.yaml`; sin tocar el código.

---

## 🛠️ Requisitos Previos

1. **Python 3.11+** (entorno conda recomendado)
2. **Pandoc** y **XeLaTeX** para la generación del PDF final:
   ```bash
   sudo apt install pandoc texlive-xetex texlive-fonts-recommended
   ```
3. Al menos un backend LLM activo (ver sección [Backends](#-backends)).

---

## 📦 Instalación

```bash
# 1. Crear entorno conda
conda create -n pdf-translator python=3.11 -y
conda activate pdf-translator

# 2. Instalar dependencias
pip install -r requirements.txt
```

**Dependencias principales:** `pymupdf4llm`, `requests`, `PyYAML`, `PyMuPDF`

---

## ⚙️ Configuración

Edita `config.yaml` para ajustar el comportamiento sin tocar el código:

```yaml
backend: "vllm"                        # "ollama" | "vllm" | "aphrodite"
api_base: "http://localhost:8000/v1"   # URL base del servidor LLM
model: "google/translategemma-4b-it"   # Nombre exacto del modelo

src: "English"      # Idioma origen
dst: "Español"      # Idioma destino

paginas: null        # null = todo el PDF, o "1-50,80-100"
chunk_tokens: 900    # Tokens por chunk de traducción
timeout: 120         # Segundos por petición al LLM
retries: 3           # Reintentos ante fallo de red
workers: 8           # Solicitudes paralelas (>1 desactiva contexto entre chunks)

output: "output"     # Carpeta de salida
no_pdf: false        # true = solo genera .md, no PDF
limpio: false        # true = ignora la caché y empieza de cero
```

---

## 🎮 Uso

```bash
# Traducir todo el PDF (configuración desde config.yaml)
python traductor.py mi_libro.pdf

# Sobrescribir parámetros puntualmente desde CLI
python traductor.py mi_libro.pdf --paginas 1-50 --workers 4
python traductor.py mi_libro.pdf --backend ollama --api-base http://localhost:11434/api
python traductor.py mi_libro.pdf --limpio   # re-traducir desde cero ignorando caché
python traductor.py mi_libro.pdf --no-pdf   # solo genera el .md, sin PDF
```

### 📂 Estructura de Salida

```
output/
└── mi_libro/
    ├── images/                    # Imágenes extraídas del PDF
    ├── mi_libro_traducido.md      # Markdown traducido (intermedio)
    ├── mi_libro_traducido.pdf     # PDF final con tipografía
    └── mi_libro_progreso.json     # Caché de progreso (se borra al terminar)
```

---

## 🔌 Backends

### Ollama (CPU/GPU, fácil de usar)

```bash
# Instalar y arrancar
ollama serve
ollama pull translategemma:12b

# config.yaml
backend: "ollama"
api_base: "http://localhost:11434/api"
model: "translategemma:12b"
workers: 1   # Ollama no tiene batching nativo; usar 1
```

### vLLM (GPU, recomendado para velocidad)

```bash
# Arrancar servidor vLLM
vllm serve google/translategemma-4b-it --port 8000

# config.yaml
backend: "vllm"
api_base: "http://localhost:8000/v1"
model: "google/translategemma-4b-it"
workers: 8   # vLLM maneja múltiples peticiones en paralelo
```

> Ver la [Guía de Migración Ollama → vLLM](MIGRATION_OLLAMA_TO_VLLM.md) para instrucciones detalladas.

### Aphrodite Engine (alternativa a vLLM)

```bash
# config.yaml
backend: "aphrodite"
api_base: "http://localhost:2242/v1"
model: "nombre-del-modelo"
```

---

## 🤖 Modelos Recomendados

| Modelo | Backend | VRAM | Calidad |
|--------|---------|------|---------|
| `google/translategemma-4b-it` | vLLM | ~8 GB | ⭐⭐⭐⭐⭐ (especializado en traducción) |
| `translategemma:12b` | Ollama | ~16 GB | ⭐⭐⭐⭐⭐ |
| `gemma3:12b` | Ollama | ~16 GB | ⭐⭐⭐⭐ |
| `llama3.1:8b` | Ollama/vLLM | ~10 GB | ⭐⭐⭐ |

---

## ⚠️ Notas Importantes

- **PDFs basados en imágenes**: Si el PDF no tiene texto seleccionable (solo imágenes escaneadas), primero debes aplicar OCR con Tesseract u otra herramienta para generar un PDF con texto embebido.
- **Tesseract**: `pymupdf4llm` puede intentar usar Tesseract internamente para análisis de layout. Si el PDF ya tiene texto (como los `_final_ocr.pdf`), esto no es necesario y se desactiva automáticamente con `use_ocr=False` en el código.
- **workers > 1**: En modo paralelo, el contexto entre chunks se desactiva porque el orden de ejecución de los threads no está garantizado. Para traducciones donde la coherencia entre párrafos es crítica, usa `workers: 1`.
