import json
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import time
from google import genai
import requests
import math 
from typing import List
import base64
from gradio_client import Client
from groq import Groq
from rank_bm25 import BM25Okapi
import cohere
import re
import string

load_dotenv()

app = FastAPI(title="Microservicio IA - GeckOS")


SYSTEM_PROMPT = """
Eres el núcleo de Inteligencia Artificial de GeckOS, un entorno virtual enfocado en potenciar el aprendizaje académico y la productividad de los estudiantes.
Tu objetivo principal es actuar como un tutor experto, resolviendo dudas académicas, y como un guía del sistema para ayudar al usuario a utilizar las herramientas disponibles.

REGLA DE COMPORTAMIENTO ESTRICTA: Eres un asistente puramente conversacional y consultivo. NO tienes la capacidad de abrir aplicaciones, crear notas, generar imágenes directamente en este chat o interactuar con el sistema de archivos del usuario. NUNCA ofrezcas guardar información, crear recordatorios o abrir herramientas. Tu deber es GUIAR al usuario paso a paso para que él mismo lo haga.

MANUAL DE INTERFAZ DE GECKOS (Usa estas instrucciones para guiar al usuario cuando pregunte cómo usar el sistema):
1. Generar Imágenes con IA: Indícale que presione el menú de las 3 rayitas en la esquina inferior izquierda, luego el botón de Configuración (engranaje), seleccione el apartado "Generar con IA" y finalmente escriba ahí su prompt.
2. Búsqueda Semántica: Indícale que entre a la aplicación "Mi Equipo", escriba su consulta en la barra de búsqueda y presione el botón de la derecha con ícono de estrella (para buscar por relevancia).
3. Análisis de Documentos: Indícale que primero seleccione y abra el archivo que desea analizar, presione el botón que dice "IA" y escriba o seleccione la acción a realizar (traducir, mejorar redacción, sacar ideas, etc.).

REGLA DE CONTEXTO EDUCATIVO: Cuando el usuario te haga una pregunta conceptual o técnica (ej. "¿Qué es un socket?", "¿Cómo funciona un bucle for?", "¿Qué es una API?"), debes responder de forma clara, con analogías si es posible, y enfocándote en que el estudiante comprenda el tema a la perfección.

DEBES responder ÚNICAMENTE con un objeto JSON válido con esta estructura simplificada exacta:
{
    "mensaje": "Tu respuesta amigable, la explicación del concepto o los pasos a seguir de forma clara."
}

Ejemplo 1 (Pregunta Educativa/Técnica):
Usuario: "¿Qué es un socket?"
Respuesta: {"mensaje": "Un socket es un punto final (endpoint) en una red de comunicación bidireccional entre dos programas. Imagínalo como una 'puerta' por donde entra y sale información entre un cliente y un servidor a través de una dirección IP y un puerto."}

Ejemplo 2 (Intento de comando del usuario / Guía de Interfaz):
Usuario: "Genera una imagen de un circuito para mi tarea"
Respuesta: {"mensaje": "Como soy un tutor conversacional, no puedo generar la imagen directamente aquí. Pero puedes hacerlo tú mismo: presiona el menú de las 3 rayitas en la esquina inferior izquierda, ve a Configuración (el engranaje), entra a 'Generar con IA' y escribe lo que necesitas. ¡Espero que te quede genial para tu tarea!"}

Ejemplo 3 (Intento de comando del usuario para guardar):
Usuario: "Anota que mañana tengo examen de redes"
Respuesta: {"mensaje": "¡Mucho éxito en tu examen de redes! Recuerda que soy un tutor virtual, por lo que te sugiero anotar tu recordatorio en tu aplicación de tareas o calendario personal. ¿Hay algún concepto de redes que quieras repasar ahora mismo?"}

Ejemplo 4 (Conversación general):
Usuario: "Hola, estoy listo para estudiar"
Respuesta: {"mensaje": "¡Esa es la actitud! ¿Con qué materia o tema te puedo ayudar a estudiar hoy?"}
"""

class ChatRequest(BaseModel):
    mensaje: str
    test_mode: bool = False

# --- ENDPOINT CHAT ---
@app.post("/chat")
def chat(req: ChatRequest):
    inicio = time.time()

    # ==========================================
    # INTERCEPTOR DE PRUEBAS DE CARGA (MOCK)
    # ==========================================
    if req.test_mode:
        time.sleep(1.5) # Simulamos 1.5 segundos de latencia de red de Google
        fin = time.time()
        return {
            "respuesta": {
                "mensaje": "[MODO PRUEBA] Hola, soy tu tutor virtual. Esta es una respuesta simulada para no consumir cuota de la API."
            },
            "metricas": {
                "tiempo_respuesta_ms": int((fin - inicio) * 1000)
            }
        }

    # ==========================================
    # FLUJO NORMAL (PRODUCCIÓN)
    # ==========================================
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return {"error": "GOOGLE_API_KEY no encontrada en .env"}

    client = genai.Client(api_key=api_key)
    prompt = f"{SYSTEM_PROMPT}\nUsuario: {req.mensaje}"

    max_reintentos = 3
    for intento in range(max_reintentos):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=dict(
                    response_mime_type="application/json",
                    temperature=0.2 
                )
            )
            
            respuesta_ia_json = json.loads(response.text)
            fin = time.time()

            return {
                "respuesta": respuesta_ia_json,
                "metricas": {
                    "tiempo_respuesta_ms": int((fin - inicio) * 1000)
                }
            }

        except Exception as e:
            if intento == max_reintentos - 1:
                return {
                    "error": "Servidores de Google ocupados tras varios intentos",
                    "detalle": str(e)
                }
            time.sleep(2)

class FondoRequest(BaseModel):
    descripcion: str
    test_mode: bool = False

# --- ENDPOINT FONDOS ---
@app.post("/generar-fondo")
def generar_fondo(req: FondoRequest):
    inicio = time.time()

    # ==========================================
    # INTERCEPTOR DE PRUEBAS DE CARGA (MOCK)
    # ==========================================
    if req.test_mode:
        time.sleep(4.5) # Simulamos 4.5 segundos, ya que la generación de imágenes es más lenta
        fin = time.time()
        # Pixel transparente 1x1 válido
        pixel_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
        return {
            "mensaje": "Fondo generado con éxito usando MOCK-TEST",
            "imagen": f"data:image/png;base64,{pixel_base64}",
            "metricas": {
                "tiempo_respuesta_ms": int((fin - inicio) * 1000)
            }
        }

    # ==========================================
    # FLUJO NORMAL (PRODUCCIÓN)
    # ==========================================
    try:
        hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
        if not hf_api_key:
            return {"error": "HUGGINGFACE_API_KEY no encontrada en .env"}

        prompt_final = f"desktop wallpaper, 16:9, masterpiece, {req.descripcion}"
        modelo_usado = ""

        try:
            # PLAN A: Intentar con FLUX.1-dev
            client = Client("black-forest-labs/FLUX.1-dev", token=hf_api_key)
            resultado = client.predict(
                prompt=prompt_final,
                seed=0,
                randomize_seed=True,
                width=1024,
                height=576,
                guidance_scale=3.5,
                num_inference_steps=28,
                api_name="/infer"
            )
            ruta_imagen = resultado[0]
            modelo_usado = "FLUX.1-dev"

        except Exception as error_flux:
            print(f"Plan A (FLUX) falló: {error_flux}. Iniciando Plan B (ERNIE)...")
            try:
                # PLAN B: Respaldo con ERNIE-Image-Turbo
                client_fallback = Client("baidu/ERNIE-Image-Turbo", token=hf_api_key)
                resultado_fallback = client_fallback.predict(
                    prompt=prompt_final,
                    size="1376x768", 
                    seed=-1,         
                    use_pe=True,
                    api_name="/generate_image"
                )
                
                ruta_imagen = resultado_fallback[0]
                modelo_usado = "ERNIE-Image-Turbo"

            except Exception as error_ernie:
                return {
                    "error": "Todos los servicios de generación de imágenes están caídos.",
                    "detalle": f"FLUX: {str(error_flux)} | ERNIE: {str(error_ernie)}"
                }

        with open(ruta_imagen, "rb") as archivo_imagen:
            imagen_bytes = archivo_imagen.read()
        
        imagen_base64 = base64.b64encode(imagen_bytes).decode('utf-8')
        formato_datos = f"data:image/webp;base64,{imagen_base64}" 

        fin = time.time()

        return {
            "mensaje": f"Fondo generado con éxito usando {modelo_usado}",
            "imagen": formato_datos,
            "metricas": {
                "tiempo_respuesta_ms": int((fin - inicio) * 1000)
            }
        }

    except Exception as e_critico:
        return {
            "error": "Fallo crítico en el endpoint de generación de fondos",
            "detalle": str(e_critico)
        }


class ArchivoVirtual(BaseModel):
    id: str
    nombre: str
    contenido: str

class BusquedaRequest(BaseModel):
    consulta: str
    archivos: List[ArchivoVirtual]
    test_mode: bool = False

def similitud_coseno(vec1, vec2):
    dot_product = sum(x * y for x, y in zip(vec1, vec2))
    magnitude_v1 = math.sqrt(sum(x * x for x in vec1))
    magnitude_v2 = math.sqrt(sum(x * x for x in vec2))
    if magnitude_v1 == 0 or magnitude_v2 == 0:
        return 0
    return dot_product / (magnitude_v1 * magnitude_v2)

def es_texto_basura(contenido, umbral_letras=0.65, longitud_minima=5):
    texto_limpio = contenido.replace(" ", "")
    if not texto_limpio or len(texto_limpio) < longitud_minima:
        return True
    letras = sum(1 for c in texto_limpio if c.isalpha())
    proporcion = letras / len(texto_limpio)
    return proporcion < umbral_letras

def es_texto_legible(contenido, min_palabras=2, min_vocales=3):
    vocales = sum(1 for c in contenido.lower() if c in 'aeiou')
    palabras = re.findall(r'[a-zA-Záéíóúñ]{2,}', contenido)
    return len(palabras) >= min_palabras and vocales >= min_vocales

def es_contenido_valido(contenido):
    if es_texto_basura(contenido):
        return False
    if not es_texto_legible(contenido):
        return False
    return True

def es_texto_visible(contenido: str) -> bool:
    if not contenido or not isinstance(contenido, str):
        return False
    # Remueve espacios, tabs, saltos de línea y caracteres no imprimibles
    limpio = ''.join(c for c in contenido if c.isprintable()).strip()
    if not limpio:
        return False
    # Al menos un carácter alfabético o numérico
    return any(c.isalnum() for c in limpio)

@app.post("/buscar")
def buscar_archivos(req: BusquedaRequest):
    inicio = time.time()

    if req.test_mode:
        time.sleep(1.0) 
        return {
            "mensaje": "Búsqueda híbrida completada usando MOCK",
            "resultados": [],
            "metricas": {"tiempo_respuesta_ms": 1000}
        }
    
    try:
        # ==========================================
        # FILTRO DE ARCHIVOS BASURA
        # ==========================================
        archivos_limpios = []
        for a in req.archivos:
        # 1) Contenido visible (no vacío, no solo símbolos)
            if not es_texto_visible(a.contenido):
                print(f"[FILTRO] Descartado archivo vacío/invisible: {a.nombre}")
                continue
        # 2) Validación de legibilidad (la que ya tienes)
            if not es_contenido_valido(a.contenido):
                print(f"[FILTRO] Descartado por basura/ilegible: {a.nombre}")
                continue
            archivos_limpios.append(a)

        if not archivos_limpios:
            return {
                "mensaje": "No se encontraron archivos con contenido legible o los archivos estaban vacíos.",
                "resultados": [],
                "metricas": {"tiempo_respuesta_ms": int((time.time() - inicio) * 1000)}
            }

        # A partir de aquí, SIEMPRE usamos 'archivos_limpios', no 'req.archivos'

        # ==========================================
        # FASE 1: ANÁLISIS LÉXICO (BM25)
        # ==========================================
        corpus_tokenizado = [f"{a.nombre} {a.contenido}".lower().split(" ") for a in archivos_limpios]  # <-- CORREGIDO
        bm25 = BM25Okapi(corpus_tokenizado)
        consulta_tokenizada = req.consulta.lower().split(" ")

        puntajes_bm25 = list(bm25.get_scores(consulta_tokenizada))
        max_bm25 = max(puntajes_bm25) if len(puntajes_bm25) > 0 and max(puntajes_bm25) > 0 else 1.0

        puntajes_bm25_norm = [score / max_bm25 for score in puntajes_bm25]

        # ==========================================
        # FASE 2: ANÁLISIS SEMÁNTICO (CON FALLBACK)
        # ==========================================
        textos_a_vectorizar = [req.consulta] + [f"{a.nombre}. {a.contenido}" for a in archivos_limpios]
        vectores = []
        modelo_usado = ""

        try:
            # --- PLAN A: GEMINI ---
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise Exception("GOOGLE_API_KEY no encontrada")

            client = genai.Client(api_key=api_key)
            respuesta_gemini = client.models.embed_content(
                model="gemini-embedding-001",
                contents=textos_a_vectorizar
            )
            vectores = [emb.values for emb in respuesta_gemini.embeddings]
            modelo_usado = "Gemini embedding-001"

        except Exception as error_gemini:
            print(f"Plan A (Gemini) falló: {error_gemini}. Activando Plan B (Cohere)...")
            try:
                # --- PLAN B: COHERE ---
                cohere_api_key = os.getenv("COHERE_API_KEY")
                if not cohere_api_key:
                    raise Exception("COHERE_API_KEY no encontrada en .env")

                co = cohere.Client(cohere_api_key)

                # Vectorizar consulta
                resp_consulta = co.embed(
                    texts=[req.consulta],
                    model='embed-multilingual-v3.0',
                    input_type='search_query'
                )
                vector_consulta = resp_consulta.embeddings[0]

                # Vectorizar archivos LIMPIOS (corregido también)
                textos_archivos = [f"{a.nombre.replace('.txt', '')}. {a.contenido}" for a in archivos_limpios]  # <-- USAMOS archivos_limpios
                resp_archivos = co.embed(
                    texts=textos_archivos,
                    model='embed-multilingual-v3.0',
                    input_type='search_document'
                )
                vectores_archivos = resp_archivos.embeddings

                vectores = [vector_consulta] + vectores_archivos
                modelo_usado = "Fallback: Cohere embed-multilingual-v3.0"

            except Exception as error_cohere:
                return {
                    "error": "Caída total de servicios de búsqueda semántica (Gemini y Cohere).",
                    "detalle": f"Gemini: {str(error_gemini)} | Cohere: {str(error_cohere)}"
                }

        # Separar vector de consulta y vectores de archivos
        vector_consulta = vectores[0]
        vectores_archivos = vectores[1:]

        # ==========================================
        # FASE 3: FUSIÓN HÍBRIDA Y FILTRO DE RELEVANCIA
        # ==========================================
        peso_lexico = 0.3
        peso_semantico = 0.7
        resultados = []

        for i, archivo in enumerate(archivos_limpios):
            similitud_semantica = similitud_coseno(vector_consulta, vectores_archivos[i])
            similitud_semantica_norm = math.pow(max(0.0, similitud_semantica), 2)

            puntaje_final = (puntajes_bm25_norm[i] * peso_lexico) + (similitud_semantica_norm * peso_semantico)
            porcentaje_final = round(puntaje_final * 100, 2)

            if porcentaje_final >= 35.0:
                resultados.append({
                    "id": archivo.id,
                    "nombre": archivo.nombre,
                    "relevancia": porcentaje_final
                })

        resultados_ordenados = sorted(resultados, key=lambda x: x["relevancia"], reverse=True)
        fin = time.time()

        return {
            "mensaje": f"Búsqueda híbrida completada usando {modelo_usado}",
            "resultados": resultados_ordenados,
            "metricas": {
                "tiempo_respuesta_ms": int((fin - inicio) * 1000)
            }
        }

    except Exception as e:
        return {
            "error": "Error interno en el endpoint de búsqueda",
            "detalle": str(e)
        }
    

class AnalisisRequest(BaseModel):
    texto: str
    accion: str 
    test_mode: bool = False

@app.post("/analizar-documento")
def analizar_documento(req: AnalisisRequest):
    inicio = time.time()

    if req.test_mode:
        time.sleep(0.8) 
        return {
            "mensaje": f"Análisis completado: {req.accion} (MOCK)",
            "modelo_ejecucion": "MOCK-LLM", 
            "respuesta": {"resultado": "Texto procesado de prueba"},
            "metricas": {"tiempo_respuesta_ms": 800}
        }
    
    prompt_analisis = f"""
    Eres el procesador de texto avanzado (la "navaja suiza") del sistema GeckOS.
    Tu tarea es interpretar y aplicar EXACTAMENTE la siguiente instrucción: '{req.accion}' sobre el texto proporcionado.
    
    Eres capaz de realizar cualquier tarea de lenguaje natural: resumir, extraer ideas principales, extender textos, mejorar la redacción, traducir, cambiar el tono, etc. Hazlo con la mayor calidad profesional posible.

    DEBES responder ÚNICAMENTE con un objeto JSON válido con esta estructura exacta:
    {{
        "resultado": "Aquí va el texto procesado final"
    }}

    Texto a procesar:
    {req.texto}
    """

    modelo_usado = ""
    respuesta_ia_json = {}

    # ==========================================
    # ENRUTADOR INTELIGENTE (ROUTER)
    # ==========================================
    # Convertimos la acción a minúsculas para buscar coincidencias fácilmente
    accion_lower = req.accion.lower()
    palabras_traduccion = ["traducir", "traduce", "traducción", "translate", "idioma", "inglés", "english"]
    
    # Si la acción contiene alguna de las palabras clave, forzamos el uso de Gemini
    forzar_gemini = any(palabra in accion_lower for palabra in palabras_traduccion)

    if not forzar_gemini:
        # ==========================================
        # RUTA 1: TAREAS GENERALES -> GROQ (Velocidad)
        # ==========================================
        try:
            groq_api_key = os.getenv("GROQ_API_KEY")
            if not groq_api_key:
                raise ValueError("GROQ_API_KEY no encontrada")

            cliente_groq = Groq(api_key=groq_api_key)
            response = cliente_groq.chat.completions.create(
                messages=[{"role": "system", "content": prompt_analisis}],
                model="llama-3.1-8b-instant", 
                response_format={"type": "json_object"} 
            )

            respuesta_ia_json = json.loads(response.choices[0].message.content)
            modelo_usado = "Llama-3.1-8b (Groq)"

        except Exception as error_groq:
            print(f"Groq falló ({error_groq}). Redirigiendo a Gemini...")
            forzar_gemini = True # Si Groq falla, activamos el switch para que el bloque de Gemini lo rescate

    # OJO: Usamos 'if' y no 'else' porque si Groq falló arriba, forzar_gemini ahora es True
    if forzar_gemini:
        # ==========================================
        # RUTA 2: TRADUCCIONES O FALLBACK -> GEMINI (Precisión)
        # ==========================================
        try:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY no encontrada")

            client = genai.Client(api_key=api_key)
            response_gemini = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt_analisis,
                config=dict(
                    response_mime_type="application/json",
                    temperature=0.2
                )
            )
            respuesta_ia_json = json.loads(response_gemini.text)
            
            # Etiquetamos dinámicamente si fue por enrutamiento o por emergencia
            if "traduc" in accion_lower or "inglés" in accion_lower or "english" in accion_lower:
                modelo_usado = "Gemini 2.5 Flash (Traductor Dedicado)"
            else:
                modelo_usado = "Gemini 2.5 Flash (Fallback)"

        except Exception as error_gemini:
            return {
                "error": "Los servidores de análisis están experimentando alta demanda.",
                "detalle": f"Error final en Gemini: {str(error_gemini)}"
            }

    fin = time.time()

    return {
        "mensaje": f"Análisis completado: {req.accion}",
        "modelo_ejecucion": modelo_usado, 
        "respuesta": respuesta_ia_json,
        "metricas": {
            "tiempo_respuesta_ms": int((fin - inicio) * 1000)
        }
    }