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

load_dotenv()

app = FastAPI(title="Microservicio IA - GeckOS")

SYSTEM_PROMPT = """
Eres el núcleo de Inteligencia Artificial de GeckOS, un sistema operativo virtual diseñado para potenciar la productividad y el aprendizaje académico.
Tu objetivo es doble: analizar lo que pide el usuario para ejecutar comandos en el sistema, y actuar como un tutor experto para resolver dudas académicas, técnicas o de desarrollo de software.

REGLA DE CONTEXTO EDUCATIVO: Si el usuario te hace una pregunta conceptual, académica o técnica (ej. "¿Qué es un socket?", "¿Cómo funciona un bucle for?", "¿Qué es la mitosis?"), DEBES responderla de forma clara, didáctica y concisa. En estos casos, el comando será "ninguno" a menos que el usuario explícitamente te pida anotar o guardar la explicación.

DEBES responder ÚNICAMENTE con un objeto JSON válido con esta estructura exacta:
{
    "mensaje": "Tu respuesta amigable o la explicación del concepto",
    "comando": "abrir" o "crear_nota" o "ninguno",
    "apps": ["lista", "de", "apps", "solo", "si", "el", "comando", "es", "abrir"],
    "contenido_nota": "El texto a guardar, solo si el comando es crear_nota. Vacío en otros casos."
}

Las aplicaciones y herramientas disponibles en el sistema son: "notas", "calculadora", "navegador", "tareas", "terminal", "archivos", "generador_fondos", "buscador_semantico", "editor_documentos".

Guía de comportamiento avanzado para herramientas de IA:
- Si el usuario te pide crear, generar o dibujar un fondo de pantalla o imagen, tu comando debe ser "abrir" la app "generador_fondos".
- Si el usuario quiere buscar un archivo por su significado o encontrar información perdida, tu comando debe ser "abrir" la app "buscador_semantico".
- Si el usuario te pide resumir, traducir, corregir o analizar un texto largo, tu comando debe ser "abrir" la app "editor_documentos".

Ejemplo 1 (Generar imagen):
Usuario: "Quiero un fondo de pantalla de un bosque nevado"
Respuesta: {"mensaje": "¡Me encanta la idea! Te abriré el Generador de Fondos para que puedas crear esa imagen.", "comando": "abrir", "apps": ["generador_fondos"], "contenido_nota": ""}

Ejemplo 2 (Buscar archivos):
Usuario: "No encuentro el pdf de la universidad, ayúdame a buscarlo"
Respuesta: {"mensaje": "¡No te preocupes! Abriendo el Buscador Semántico Inteligente para encontrar ese documento.", "comando": "abrir", "apps": ["buscador_semantico"], "contenido_nota": ""}

Ejemplo 3 (Crear Nota):
Usuario: "Anota que mañana tengo que comprar café"
Respuesta: {"mensaje": "¡Listo! He creado una nota con tu recordatorio.", "comando": "crear_nota", "apps": [], "contenido_nota": "Comprar café mañana"}

Ejemplo 4 (Pregunta Educativa/Técnica):
Usuario: "¿Qué es un socket?"
Respuesta: {"mensaje": "Un socket es un punto final (endpoint) en una red de comunicación bidireccional entre dos programas. Imagínalo como una 'puerta' por donde entra y sale información entre un cliente y un servidor a través de una dirección IP y un puerto. ¿Te gustaría que cree una nota con esta definición?", "comando": "ninguno", "apps": [], "contenido_nota": ""}
"""

class ChatRequest(BaseModel):
    mensaje: str

@app.post("/chat")
def chat(req: ChatRequest):
    inicio = time.time()
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

@app.post("/generar-fondo")
def generar_fondo(req: FondoRequest):
    inicio = time.time()
    try:
        hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
        if not hf_api_key:
            return {"error": "HUGGINGFACE_API_KEY no encontrada en .env"}

        prompt_final = f"desktop wallpaper, 16:9, masterpiece, {req.descripcion}"
        
        modelo_usado = ""

        try:
            # ==========================================
            # PLAN A: Intentar con FLUX.1-dev
            # ==========================================
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
                # ==========================================
                # PLAN B: Respaldo con ERNIE-Image-Turbo
                # ==========================================
                client_fallback = Client("baidu/ERNIE-Image-Turbo", token=hf_api_key)
                resultado_fallback = client_fallback.predict(
                    prompt=prompt_final,
                    size="1376x768", # Formato 16:9 exacto sacado de tu view_api
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

        # --- PROCESAMIENTO A BASE64 ---
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

def similitud_coseno(vec1, vec2):
    dot_product = sum(x * y for x, y in zip(vec1, vec2))
    magnitude_v1 = math.sqrt(sum(x * x for x in vec1))
    magnitude_v2 = math.sqrt(sum(x * x for x in vec2))
    if magnitude_v1 == 0 or magnitude_v2 == 0:
        return 0
    return dot_product / (magnitude_v1 * magnitude_v2)

@app.post("/buscar")
def buscar_archivos(req: BusquedaRequest):
    inicio = time.time()
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return {"error": "GOOGLE_API_KEY no encontrada en .env"}

        client = genai.Client(api_key=api_key)
        
        texto_consulta = f"Intención de búsqueda del usuario: {req.consulta}"
        
        respuesta_consulta = client.models.embed_content(
            model="gemini-embedding-001",
            contents=req.consulta
        )
        vector_consulta = respuesta_consulta.embeddings[0].values

        resultados = []

        for archivo in req.archivos:
            texto_archivo = f"{archivo.nombre}. {archivo.contenido}"
            
            respuesta_archivo = client.models.embed_content(
                model="gemini-embedding-001",
                contents=texto_archivo
            )
            vector_archivo = respuesta_archivo.embeddings[0].values
            
            similitud = similitud_coseno(vector_consulta, vector_archivo)
            
            porcentaje = max(0.0, min(100.0, round((similitud - 0.5) * 200, 2)))
            
            resultados.append({
                "id": archivo.id,
                "nombre": archivo.nombre,
                "relevancia": porcentaje
            })

        resultados_ordenados = sorted(resultados, key=lambda x: x["relevancia"], reverse=True)

        fin = time.time()

        return {
            "mensaje": "Búsqueda semántica completada",
            "resultados": resultados_ordenados,
            "metricas": {
                "tiempo_respuesta_ms": int((fin - inicio) * 1000)
            }
        }

    except Exception as e:
        return {
            "error": "Fallo al realizar la búsqueda semántica",
            "detalle": str(e)
        }
class AnalisisRequest(BaseModel):
    texto: str
    accion: str 

@app.post("/analizar-documento")
def analizar_documento(req: AnalisisRequest):
    inicio = time.time()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return {"error": "GOOGLE_API_KEY no encontrada en .env"}

    client = genai.Client(api_key=api_key)
    
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

    try:
        # ==========================================
        # PLAN A: El modelo más avanzado (2.5 Flash)
        # ==========================================
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt_analisis,
            config=dict(
                response_mime_type="application/json",
                temperature=0.2
            )
        )
        respuesta_ia_json = json.loads(response.text)
        modelo_usado = "gemini-2.5-flash"

    except Exception as error_principal:
        print(f"Plan A falló ({error_principal}). Iniciando Plan B (Groq/Llama3)...")
        try:
            # ==========================================
            # PLAN B: Respaldo de ultra alta velocidad (Groq)
            # ==========================================
            groq_api_key = os.getenv("GROQ_API_KEY")
            if not groq_api_key:
                raise ValueError("GROQ_API_KEY no encontrada en .env")

            cliente_groq = Groq(api_key=groq_api_key)

            response_fallback = cliente_groq.chat.completions.create(
                messages=[
                    {"role": "system", "content": prompt_analisis}
                ],
                model="llama-3.1-8b-instant", 
                response_format={"type": "json_object"} 
            )

            respuesta_ia_json = json.loads(response_fallback.choices[0].message.content)
            modelo_usado = "Llama-3.1-8b (Groq)"

        except Exception as error_secundario:
            return {
                "error": "Los servidores de análisis están experimentando alta demanda.",
                "detalle": f"Plan A: {str(error_principal)} | Plan B: {str(error_secundario)}"
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