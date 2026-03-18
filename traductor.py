import os
import sys
import argparse
import requests
import fitz  # PyMuPDF
import json
import shutil
from pathlib import Path

# --- Variables de Configuración ---
MODELO_OLLAMA = "translategemma:12b"
IDIOMA_ORIGEN = "English"
CODIGO_ORIGEN = "EN"
IDIOMA_DESTINO = "Español"
CODIGO_DESTINO = "ES"
OLLAMA_API_BASE = "http://localhost:11434/api"

# Configuración de Fuentes (Deben ser fuentes estándar de PDF o rutas a archivos .ttf/.otf)
# Fuentes estándar de PDF: 'helv' (Helvetica), 'tiro' (Times-Roman), 'cour' (Courier)
FUENTE_NORMAL = "tiro"
FUENTE_NEGRITA = "TiBo" # Times-Bold

def validar_fuentes():
    """Valida que las fuentes elegidas sean soportadas por PyMuPDF."""
    print("[*] Validando fuentes de texto...")
    fuentes_soportadas = fitz.fitz_fontdescriptors.keys()
    
    fuentes_a_validar = [FUENTE_NORMAL, FUENTE_NEGRITA]
    
    for fuente in fuentes_a_validar:
        # Check standard fonts first
        is_standard = fuente in fuentes_soportadas or fuente.lower() in ['helv', 'tiro', 'cour', 'symb', 'zadb', 'tibo', 'hebo']
        
        # If not standard, assume it's a file path and check if the file exists
        if not is_standard:
            if not os.path.exists(fuente):
                print(f"[ERROR] La fuente especificada no es una fuente base válida ni una ruta de archivo existente: '{fuente}'")
                print("Fuentes base válidas: helv (Helvetica), tiro (Times), cour (Courier), TiBo (Times-Bold), etc.")
                sys.exit(1)
                
    print(f"[+] Fuentes OK: Normal='{FUENTE_NORMAL}', Negrita='{FUENTE_NEGRITA}'\n")

def verificar_ollama():
    """Verifica si el servicio Ollama está disponible y maneja los modelos."""
    print("[*] Verificando estado de Ollama...")
    try:
        response = requests.get("http://localhost:11434/", timeout=5)
        if response.status_code != 200:
            raise requests.exceptions.ConnectionError
    except requests.exceptions.RequestException:
        print("\n[ERROR] Ollama no está respondiendo en localhost:11434.")
        print("Asegúrate de que el servicio 'ollama serve' esté en ejecución.")
        sys.exit(1)

    print("[*] Ollama está en línea. Verificando modelos cargados...")
    
    # Obtener modelos cargados actualmente
    try:
        ps_response = requests.get(f"{OLLAMA_API_BASE}/ps", timeout=5)
        ps_data = ps_response.json()
        modelos_cargados = [m['name'] for m in ps_data.get('models', [])]
        
        for modelo in modelos_cargados:
            if modelo != MODELO_OLLAMA:
                print(f"[-] Descargando modelo no deseado de la memoria: {modelo}")
                # Descargar modelo enviando keep_alive=0
                payload = {
                    "model": modelo,
                    "keep_alive": 0
                }
                requests.post(f"{OLLAMA_API_BASE}/generate", json=payload)
    except Exception as e:
        print(f"[-] Advertencia al consultar modelos cargados: {e}")

    print(f"[*] Preparando modelo principal: {MODELO_OLLAMA}")
    # Enviar una solicitud vacía con keep_alive para precargarlo
    try:
        payload = {
            "model": MODELO_OLLAMA,
            "prompt": "",
            "keep_alive": "5m" # Mantener vivo por 5 minutos
        }
        # Hacemos una petición rápida, aunque falle por prompt vacío, forzará la carga
        requests.post(f"{OLLAMA_API_BASE}/generate", json=payload, timeout=2)
    except requests.exceptions.Timeout:
        # Es normal que tarde en cargar la primera vez
        print(f"[*] El modelo {MODELO_OLLAMA} se está cargando en memoria...")
    except Exception as e:
        print(f"[-] Advertencia al precargar modelo: {e}")
    
    print("[+] Ollama listo.\n")

def traducir_texto(texto, contexto_previo=""):
    """Envía el texto a Ollama para su traducción."""
    if not texto.strip():
        return ""
        
    context_str = ""
    if contexto_previo:
        context_str = f"Context from previous paragraph (do not translate this, use it only for context):\n{contexto_previo}\n\n"
        
    prompt = f"""Translate {CODIGO_ORIGEN} to {CODIGO_DESTINO} naturally.

RULES:
1. Return ONLY the translation. No notes.
2. DO NOT translate URLs, emails, code or unknown words.
3. If text is only symbols/numbers, return it exactly as is.
4. Keep the exact same meaning.

{context_str}Text to translate:
{texto}"""
    
    payload = {
        "model": MODELO_OLLAMA,
        "prompt": prompt,
        "stream": False
    }
    
    try:
        response = requests.post(f"{OLLAMA_API_BASE}/generate", json=payload, timeout=60)
        response.raise_for_status()
        texto_traducido = response.json().get("response", "").strip()
        
        # Eliminar puntos suspensivos al final si el original no los tenía
        if texto_traducido.endswith("...") and not texto.strip().endswith("..."):
            texto_traducido = texto_traducido[:-3].strip()
            
        return texto_traducido
    except Exception as e:
        print(f"\n[ERROR] Falla al traducir con Ollama: {e}")
        return texto # En caso de fallo, devolvemos el original para no perder el bloque

def procesar_pagina(pdf_path, temp_dir, num_pagina, contexto_previo=""):
    """Extrae, traduce y reemplaza el texto de una página específica."""
    page_filename = f"page_{num_pagina:04d}.pdf"
    page_tr_filename = f"page_tr_{num_pagina:04d}.pdf"
    page_path = os.path.join(temp_dir, page_filename)
    page_tr_path = os.path.join(temp_dir, page_tr_filename)
    
    # 1. Separar la página si no existe
    if not os.path.exists(page_path):
        doc = fitz.open(pdf_path)
        page_doc = fitz.open()
        page_doc.insert_pdf(doc, from_page=num_pagina, to_page=num_pagina)
        page_doc.save(page_path)
        page_doc.close()
        doc.close()
    
    contexto_actual = contexto_previo
    
    # 2. Si ya está traducida, saltar
    if os.path.exists(page_tr_path):
        print(f"  [>] Página {num_pagina + 1} ya procesada (skipping).")
        return page_tr_path, contexto_actual
        
    print(f"  [>] Procesando y traduciendo página {num_pagina + 1}...")
    
    # 3. Procesar la página individual
    doc_page = fitz.open(page_path)
    page = doc_page[0]
    
    # Extraer bloques de texto (dict) para tener las coordenadas
    dict_text = page.get_text("dict")
    
    for block in dict_text.get("blocks", []):
        if block.get("type") == 0:  # Tipo 0 es texto
            # Recolectar el texto de todo el bloque para mantener el contexto
            texto_original = ""
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    texto_original += span.get("text", "") + " "
                texto_original += "\n"
                
            texto_original = texto_original.strip()
            
            # Solo traducir si el bloque contiene al menos una letra (evita enviar solo números o símbolos)
            if any(c.isalpha() for c in texto_original):
                bbox = fitz.Rect(block["bbox"])
                
                # Traducir texto completo del bloque
                texto_traducido = traducir_texto(texto_original, contexto_actual)
                
                if texto_traducido:
                    contexto_actual = texto_traducido
                    # Añadir anotación de redacción (borrado) para todo el bloque
                    page.add_redact_annot(bbox)
                    page.apply_redactions() # Aplicar borrado físico
                    
                    # Intentar obtener el tamaño de fuente y estilo del primer span del bloque
                    es_negrita = False
                    es_titulo = False
                    try:
                        primer_span = block["lines"][0]["spans"][0]
                        fontsize_orig = primer_span.get("size", 11)
                        fuente_original = primer_span.get("font", "").lower()
                        
                        # Comparamos el tamaño del span actual para evitar que "contamine"
                        # Muchos libros empiezan párrafos con Letras Capitales grandes o un par de palabras en negrita.
                        # Consideramos "Título" SÓLO si es significativamente más grande que 16
                        if fontsize_orig >= 16:
                            es_titulo = True
                            es_negrita = True
                        # Asignamos negrita si el nombre de la fuente claramente lo indica, PERO SOLO si el tamaño es > 12
                        # para evitar que párrafos enteros con falsos positivos de metadatos se pongan en negrita.
                        elif any(x in fuente_original for x in ["bold", "black", "heavy"]) and "regular" not in fuente_original and fontsize_orig > 12:
                            es_negrita = True
                    except:
                        pass
                        
                    # Seleccionar la fuente final y estandarizar tamaños
                    if es_titulo:
                        font = FUENTE_NEGRITA
                        fontsize = 16
                        alineacion = fitz.TEXT_ALIGN_CENTER # Los títulos lucen mejor centrados
                    elif es_negrita:
                        font = FUENTE_NEGRITA
                        fontsize = 11
                        alineacion = fitz.TEXT_ALIGN_LEFT # Subtítulos menores
                    else:
                        font = FUENTE_NORMAL
                        fontsize = 11
                        alineacion = fitz.TEXT_ALIGN_JUSTIFY # Texto regular formal y justificado
                    
                    # Intentar insertar ajustando el texto al rectángulo reduciendo la fuente si no cabe
                    try:
                        rc = -1
                        # Bucle para reducir el tamaño de la fuente hasta que quepa (mínimo tamaño 4)
                        while fontsize >= 4 and rc < 0:
                            rc = page.insert_textbox(bbox, texto_traducido, fontsize=fontsize, fontname=font, align=alineacion)
                            if rc < 0:
                                fontsize -= 1
                                
                        # Si aún en tamaño 4 no cabe, insertamos forzosamente sin textbox para al menos ver el texto
                        if rc < 0:
                            page.insert_text(bbox.top_left, texto_traducido, fontsize=6, fontname=font)
                    except Exception as e:
                        print(f"Error insertando texto: {e}")
                                
    # Guardar la página modificada
    doc_page.save(page_tr_path)
    doc_page.close()
    
    return page_tr_path, contexto_actual

def main():
    parser = argparse.ArgumentParser(description="Traductor de PDF usando Ollama")
    parser.add_argument("archivo", help="Ruta al archivo PDF a traducir")
    args = parser.parse_args()
    
    archivo_pdf = args.archivo
    
    if not os.path.exists(archivo_pdf):
        print(f"[ERROR] El archivo '{archivo_pdf}' no existe.")
        sys.exit(1)
        
    validar_fuentes()
    verificar_ollama()
    
    nombre_base = Path(archivo_pdf).stem
    output_dir = Path("output") / nombre_base
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir / "temp"
    temp_dir.mkdir(exist_ok=True)
    
    archivo_final = output_dir / f"{nombre_base}_traducido.pdf"
    
    if os.path.exists(archivo_final):
        print(f"[*] El archivo final '{archivo_final}' ya existe. Traducción completada previamente.")
        sys.exit(0)
        
    print(f"[*] Procesando el archivo: {archivo_pdf}")
    print(f"[*] Carpeta de salida: {output_dir}")
    print(f"[*] Traducción: {IDIOMA_ORIGEN} -> {IDIOMA_DESTINO}")
    
    # Abrir documento principal para saber cuántas páginas tiene
    try:
        doc_principal = fitz.open(archivo_pdf)
        total_paginas = len(doc_principal)
        doc_principal.close()
    except Exception as e:
        print(f"[ERROR] No se pudo leer el archivo PDF: {e}")
        sys.exit(1)
        
    print(f"[*] Total de páginas a procesar: {total_paginas}")
    
    paginas_traducidas_paths = []
    contexto = ""
    
    for i in range(total_paginas):
        path_traducida, contexto = procesar_pagina(archivo_pdf, temp_dir, i, contexto)
        paginas_traducidas_paths.append(path_traducida)
        
    print(f"[*] Todas las {total_paginas} páginas han sido procesadas.")
    print(f"[*] Uniendo páginas traducidas en archivo final...")
    
    doc_final = fitz.open()
    for tr_path in paginas_traducidas_paths:
        page_doc = fitz.open(tr_path)
        doc_final.insert_pdf(page_doc)
        page_doc.close()
        
    doc_final.save(archivo_final)
    doc_final.close()
    
    print(f"[+] Archivo final guardado exitosamente en: {archivo_final}")
    
    print(f"[*] Limpiando archivos temporales...")
    try:
        shutil.rmtree(temp_dir)
        print("[+] Limpieza completada.")
    except Exception as e:
        print(f"[-] Advertencia al limpiar temporales: {e}")
        
    print("\n[+] ¡PROCESO TERMINADO CON ÉXITO!")

if __name__ == "__main__":
    main()
