# Traductor de PDF Local con IA (Ollama)

Un script en Python diseñado para traducir documentos PDF completos de forma local, página por página, manteniendo el diseño original y utilizando modelos de lenguaje (LLMs) ejecutados en tu propia máquina a través de [Ollama](https://ollama.com).

## 🚀 Características
- **Traducción 100% Local y Privada**: Sin APIs de pago ni envío de datos a terceros. Funciona completamente off-line gracias a Ollama.
- **Gestión Automática del Modelo**: Detecta si Ollama está en ejecución, descarga modelos innecesarios de la memoria VRAM y precarga el modelo objetivo (por defecto `translategemma:12b`).
- **Resiliencia y Reanudación**: Divide el PDF y procesa página por página guardando temporales. Si el proceso se interrumpe, continuará exactamente donde se quedó.
- **Contexto Persistente**: La IA recuerda el último párrafo traducido para proporcionar contexto a la siguiente oración, evitando perder concordancia en textos divididos.
- **Formateo Inteligente**: Detecta títulos y texto regular. Usa fuentes y tamaños estandarizados (Centrado/16pt para títulos con negritas, Justificado/11pt para el resto), y reajusta automáticamente el tamaño si la traducción es demasiado larga para la caja.
- **Limpieza Automática**: Combina todas las páginas en el nuevo PDF final y elimina los temporales de manera automática.

## 🛠️ Requisitos Previos

1. **Ollama**: Debes tener instalado Ollama y haber descargado el modelo que usarás.
   ```bash
   ollama run translategemma:12b
   ```
2. **Python 3.10+**: O un gestor de entornos como Conda.

## 📦 Instalación

1. Clona este repositorio o descarga los archivos.
2. Es sumamente recomendable crear un entorno virtual para las dependencias (usando `venv` o `conda`):
   ```bash
   conda create -n pdf-translator python=3.11 -y
   conda activate pdf-translator
   ```
3. Instala los requerimientos a través de pip:
   ```bash
   pip install -r requirements.txt
   ```
   *Dependencias Principales: `PyMuPDF` (para el procesado PDF) y `requests` (para la API de Ollama).*

## ⚙️ Configuración (Opcional)

Si deseas traducir a un idioma distinto o utilizar otro modelo, puedes editar las variables globales al inicio de `traductor.py`:

```python
MODELO_OLLAMA = "translategemma:12b"
IDIOMA_ORIGEN = "English"
CODIGO_ORIGEN = "EN"
IDIOMA_DESTINO = "Español"
CODIGO_DESTINO = "ES"
FUENTE_NORMAL = "tiro"      # Times-Roman
FUENTE_NEGRITA = "TiBo"     # Times-Bold
```

## 🎮 Uso

Asegúrate de que Ollama se esté ejecutando en segundo plano en tu equipo (`ollama serve`).

Luego, simplemente pasa la ruta de tu PDF como argumento al script:

```bash
python traductor.py mi_documento.pdf
```

### 📂 Estructura de Salida
El script creará automáticamente una carpeta llamada `output` en el directorio de trabajo:
- `output/mi_documento/temp/`: Almacena las páginas divididas para evitar procesar toda la memoria RAM a la vez. (Se borra al finalizar).
- `output/mi_documento/mi_documento_traducido.pdf`: El archivo final y unificado.

## ⚠️ Limitaciones Conocidas

- **Bleeding de Negritas**: En algunos libros o PDFs mal formateados (especialmente escaneos pasados por OCR), los metadatos de las fuentes del título no se cierran de forma limpia. El script cuenta con validadores para ignorar falsos positivos, pero en casos extremos todo un bloque podría renderizarse en negritas.
- **PDFs basados en Imágenes**: El script extrae el código del texto. Si tu PDF está compuesto literalmente de fotografías de páginas escaneadas sin una capa de OCR integrada, el script no encontrará texto que extraer.
