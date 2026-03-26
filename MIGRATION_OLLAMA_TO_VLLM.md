# Guía de Migración: Ollama → vLLM

Esta guía documenta todos los cambios necesarios para migrar un script de traducción (o cualquier script que use un LLM local) de **Ollama** a **vLLM**. Está basada en la migración real de `traductor.py`.

---

## ¿Por qué migrar a vLLM?

| Característica | Ollama | vLLM |
|---|---|---|
| API compatible con OpenAI | ✅ `/api/chat` (formato propio) | ✅ `/v1/chat/completions` (estándar) |
| Batching de peticiones | ❌ Una a la vez | ✅ Múltiples en paralelo (PagedAttention) |
| Throughput (peticiones/s) | Bajo | Alto (3-10x más rápido en GPU) |
| Configuración | Simple | Requiere más parámetros al arrancar |
| Soporte de modelos HuggingFace | Indirecto (via Modelfile) | ✅ Directo (`google/translategemma-4b-it`) |
| Workers paralelos | ❌ Sin ganancia real | ✅ Escala linealmente hasta saturar GPU |

---

## 1. Arranque del servidor

### Ollama
```bash
ollama serve
ollama pull translategemma:12b   # descarga el modelo
```

### vLLM
```bash
pip install vllm

# Arrancar con un modelo de HuggingFace
vllm serve google/translategemma-4b-it --port 8000

# Opciones útiles:
vllm serve google/translategemma-4b-it \
    --port 8000 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --dtype bfloat16
```

---

## 2. Diferencias de API

### Endpoint

| | Ollama | vLLM |
|---|---|---|
| Chat | `POST /api/chat` | `POST /v1/chat/completions` |
| Listar modelos | `GET /api/tags` | `GET /v1/models` |
| Base URL | `http://localhost:11434/api` | `http://localhost:8000/v1` |

### Formato del payload

**Ollama:**
```python
payload = {
    "model": "translategemma:12b",
    "messages": [
        {"role": "system", "content": "You are a translator..."},
        {"role": "user",   "content": "Translate: ..."},
    ],
    "stream": False,
    "options": {"temperature": 0.3},
}
r = requests.post("http://localhost:11434/api/chat", json=payload)
traduccion = r.json()["message"]["content"]
```

**vLLM (OpenAI-compatible):**
```python
payload = {
    "model": "google/translategemma-4b-it",
    "messages": [
        {"role": "user", "content": "Translate: ..."},
    ],
    "temperature": 0.3,
    "max_tokens": 4096,
}
r = requests.post("http://localhost:8000/v1/chat/completions", json=payload)
traduccion = r.json()["choices"][0]["message"]["content"]
```

#### Diferencias clave:
- `options.temperature` → `temperature` (nivel raíz)
- `stream: false` → no es necesario (default)
- `r.json()["message"]["content"]` → `r.json()["choices"][0]["message"]["content"]`
- vLLM acepta `max_tokens` directamente en el payload

---

## 3. Modelos y nombres

En Ollama los modelos usan el nombre del Modelfile (`translategemma:12b`).  
En vLLM se usa el identificador de HuggingFace (`google/translategemma-4b-it`).

Para verificar qué modelos están cargados:

```bash
# Ollama
curl http://localhost:11434/api/tags

# vLLM
curl http://localhost:8000/v1/models
```

---

## 4. Modelos con formato de prompt estricto (TranslateGemma)

`google/translategemma-4b-it` requiere un formato de marcas **específico** en el mensaje del usuario:

```
<<<source>>>{codigo_iso_origen}<<<target>>>{codigo_iso_destino}<<<text>>>{texto_a_traducir}
```

**Ejemplo:**
```python
user_msg = f"<<<source>>>en<<<target>>>es<<<text>>>{texto}"

payload = {
    "model": "google/translategemma-4b-it",
    "messages": [{"role": "user", "content": user_msg}],
    "temperature": 0.3,
    "max_tokens": 4096,
}
```

**Reglas importantes para TranslateGemma:**
- ❌ **No incluir `system` message** — el modelo puede rechazarlo o ignorarlo.
- ❌ **No inyectar contexto dentro de `<<<text>>>`** — el marcador debe contener solo el texto a traducir.
- ✅ Los códigos deben ser ISO 639-1 de 2 letras: `en`, `es`, `fr`, `de`, `pt`, etc.
- ✅ El modelo a veces repite los marcadores en su respuesta; añade limpieza post-generación:

```python
import re

def limpiar_respuesta(texto: str) -> str:
    # Si el modelo repite los marcadores, extraer solo la parte traducida
    match = re.search(
        r"<<<source>>>[a-z]{2}<<<target>>>[a-z]{2}<<<text>>>(.*)",
        texto, flags=re.DOTALL | re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()
    return texto.strip()
```

---

## 5. Verificación del servidor

**Ollama:**
```python
import requests

def verificar_ollama(api_base, model):
    r = requests.get(api_base.replace("/api", ""), timeout=5)
    r.raise_for_status()
    # Precargar modelo
    requests.post(f"{api_base}/generate",
                  json={"model": model, "prompt": "", "keep_alive": "10m"},
                  timeout=3)
```

**vLLM (y Aphrodite):**
```python
def verificar_vllm(api_base, model):
    r = requests.get(f"{api_base}/models", timeout=10)
    r.raise_for_status()
    modelos = [m["id"] for m in r.json().get("data", [])]
    if model not in modelos:
        raise RuntimeError(f"Modelo '{model}' no está cargado. Disponibles: {modelos}")
```

---

## 6. Peticiones en paralelo

vLLM está diseñado para manejar múltiples peticiones concurrentes gracias a **PagedAttention**. En Ollama esto no tiene ganancia real.

```python
import concurrent.futures
import threading

cache_lock = threading.Lock()

def traducir_uno(i, chunk):
    """Traduce un chunk. Thread-safe."""
    payload = {
        "model": "google/translategemma-4b-it",
        "messages": [{"role": "user", "content": f"<<<source>>>en<<<target>>>es<<<text>>>{chunk}"}],
        "temperature": 0.3,
        "max_tokens": 4096,
    }
    r = requests.post("http://localhost:8000/v1/chat/completions", json=payload, timeout=120)
    return i, r.json()["choices"][0]["message"]["content"]

chunks = [...]  # lista de textos a traducir
resultados = {}

with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    futures = {executor.submit(traducir_uno, i, c): i for i, c in enumerate(chunks)}
    for future in concurrent.futures.as_completed(futures):
        idx, texto = future.result()
        resultados[idx] = texto

# Ensamblar en orden
output = "\n\n".join(resultados[i] for i in range(len(chunks)))
```

**Workers recomendados según VRAM:**

| VRAM disponible | Workers sugeridos |
|---|---|
| 8 GB | 4–6 |
| 16 GB | 8–12 |
| 24 GB+ | 12–16 |

---

## 7. Checklist de migración

- [ ] Instalar vLLM: `pip install vllm`
- [ ] Arrancar el servidor vLLM con el modelo deseado
- [ ] Cambiar `api_base` de `http://localhost:11434/api` → `http://localhost:8000/v1`
- [ ] Cambiar endpoint de `/api/chat` → `/v1/chat/completions`
- [ ] Cambiar nombre del modelo al identificador HuggingFace
- [ ] Cambiar `r.json()["message"]["content"]` → `r.json()["choices"][0]["message"]["content"]`
- [ ] Mover `temperature` del dict `options` al nivel raíz del payload
- [ ] Añadir `max_tokens` al payload
- [ ] Si usas TranslateGemma: aplicar formato `<<<source>>>...<<<target>>>...<<<text>>>...`
- [ ] Si usas TranslateGemma: eliminar el `system` message
- [ ] Añadir limpieza de marcadores en la respuesta del modelo
- [ ] Implementar peticiones paralelas con `ThreadPoolExecutor`
- [ ] Probar con `curl` antes de ejecutar el script completo:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/translategemma-4b-it",
    "messages": [{"role":"user","content":"<<<source>>>en<<<target>>>es<<<text>>>Hello world"}],
    "temperature": 0.3,
    "max_tokens": 50
  }'
```

---

## 8. Troubleshooting

### `400 Bad Request`
- Verifica que el nombre del modelo en el payload coincida **exactamente** con el que reporta `/v1/models`.
- Para TranslateGemma: asegúrate de que el user message empieza con `<<<source>>>` y no contiene un `system` message.

### `Model not found` / `404`
- El modelo no está cargado. Verifica con `curl http://localhost:8000/v1/models`.
- Si el modelo está en proceso de carga, espera unos segundos y reintenta.

### Respuesta vacía tras limpiar
- El modelo devolvió solo los marcadores sin texto. Añade reintentos y verifica que el texto a traducir no esté vacío.

### OOM / CUDA out of memory
- Reduce `--gpu-memory-utilization` (ej: `0.85`).
- Reduce `workers` en tu script.
- Reduce `--max-model-len`.
