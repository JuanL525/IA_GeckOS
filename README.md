# Microservicio IA - GeckOS

Backend de inteligencia artificial para **GeckOS**, un entorno virtual de escritorio orientado al aprendizaje académico. Expone una API REST construida con **FastAPI** que procesa peticiones de forma independiente: no mantiene sesiones, no persiste datos de usuario y no gestiona el estado del sistema de archivos virtual. Toda la información necesaria para cada operación viaja en el cuerpo de la solicitud.

## Enlaces

| Recurso | URL |
|---|---|
| API en producción (Swagger) | https://ia-geckos.onrender.com/docs |
| Base URL del servicio | https://ia-geckos.onrender.com |
| Repositorio | https://github.com/JuanL525/IA_GeckOS |
| Documentación de la API | https://juan-lucero-s-team.docs.buildwithfern.com/ia-geck-os/chat |

## Arquitectura

Este componente es **stateless** por diseño:

- Cada endpoint recibe el contexto completo en la petición (mensaje, archivos, texto a analizar, etc.).
- No hay base de datos, caché de sesión ni almacenamiento de conversaciones.
- El frontend de GeckOS es responsable de conservar el estado del escritorio, los archivos virtuales y el historial del usuario.
- Las respuestas incluyen métricas de latencia (`tiempo_respuesta_ms`) pero no modifican ningún recurso del servidor entre llamadas.

Este modelo facilita el despliegue horizontal, la integración con balanceadores de carga y la ejecución de pruebas de carga sin efectos secundarios entre instancias.

## Capacidades

| Endpoint | Descripción | Proveedor principal | Fallback |
|---|---|---|---|
| `POST /chat` | Tutor virtual conversacional para dudas académicas y guía de interfaz | Google Gemini 2.5 Flash | Reintentos automáticos (3 intentos) |
| `POST /buscar` | Búsqueda híbrida (BM25 + similitud semántica) sobre archivos virtuales | Gemini Embedding 001 | Cohere embed-multilingual-v3.0 |
| `POST /generar-fondo` | Generación de fondos de pantalla 16:9 a partir de texto | FLUX.1-dev (Hugging Face) | ERNIE-Image-Turbo |
| `POST /analizar-documento` | Procesamiento de texto según una acción en lenguaje natural | Groq (Llama 3.1 8B) | Gemini 2.5 Flash |

Las traducciones se enrutan directamente a Gemini por precisión. Si Groq no está disponible, el análisis de documentos recurre automáticamente a Gemini.

## Requisitos

- Python 3.10 o superior
- Cuentas y claves API en los proveedores que se vayan a utilizar

## Instalación

```bash
git clone https://github.com/JuanL525/IA_GeckOS.git
cd IA_GeckOS
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

## Configuración

Cree un archivo `.env` en la raíz del proyecto con las claves necesarias:

```env
GOOGLE_API_KEY=          # Obligatoria: /chat, /buscar, /analizar-documento
HUGGINGFACE_API_KEY=     # Obligatoria: /generar-fondo
GROQ_API_KEY=            # Obligatoria: /analizar-documento (tareas generales)
COHERE_API_KEY=          # Opcional: fallback de /buscar si Gemini falla
```

## Ejecución

**Local**

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

**Producción**

El servicio está desplegado en Render. Para consumir la API en producción, use la base URL `https://ia-geckos.onrender.com`. La documentación interactiva está disponible en [ia-geckos.onrender.com/docs](https://ia-geckos.onrender.com/docs).

Referencia detallada de endpoints, ejemplos de petición y snippets en varios lenguajes: [Documentación IA GeckOS](https://juan-lucero-s-team.docs.buildwithfern.com/ia-geck-os/chat).

## API

Todos los endpoints aceptan el parámetro opcional `test_mode: true`, que devuelve respuestas simuladas sin consumir cuota de APIs externas. Útil para desarrollo, pruebas unitarias y pruebas de carga.

### POST /chat

Tutor conversacional. Responde en JSON con explicaciones académicas o instrucciones para usar las herramientas de GeckOS.

**Request**

```json
{
  "mensaje": "¿Qué es un socket?",
  "test_mode": false
}
```

**Response**

```json
{
  "respuesta": {
    "mensaje": "Un socket es un punto final en una red de comunicación..."
  },
  "metricas": {
    "tiempo_respuesta_ms": 842
  }
}
```

### POST /buscar

Recibe la consulta y la lista completa de archivos virtuales del cliente. Aplica filtrado de contenido ilegible, ranking léxico (BM25) y similitud semántica (coseno). Solo devuelve resultados con relevancia >= 20%.

**Request**

```json
{
  "consulta": "universidad",
  "archivos": [
    {
      "id": "1",
      "nombre": "clases.txt",
      "contenido": "Tengo que ir a la facultad de ingeniería."
    }
  ],
  "test_mode": false
}
```

**Response**

```json
{
  "mensaje": "Búsqueda híbrida completada usando Gemini embedding-001",
  "resultados": [
    {
      "id": "1",
      "nombre": "clases.txt",
      "relevancia": 78.42
    }
  ],
  "metricas": {
    "tiempo_respuesta_ms": 1203
  }
}
```

### POST /generar-fondo

Genera una imagen panorámica codificada en base64 (`data:image/webp;base64,...`).

**Request**

```json
{
  "descripcion": "paisaje montañoso al atardecer",
  "test_mode": false
}
```

**Response**

```json
{
  "mensaje": "Fondo generado con éxito usando FLUX.1-dev",
  "imagen": "data:image/webp;base64,...",
  "metricas": {
    "tiempo_respuesta_ms": 12450
  }
}
```

### POST /analizar-documento

Aplica una acción de lenguaje natural sobre un texto: resumir, traducir, mejorar redacción, extraer ideas, etc.

**Request**

```json
{
  "texto": "La Inteligencia Artificial es el campo de estudio que busca crear sistemas capaces de aprender.",
  "accion": "Resumir en una oración",
  "test_mode": false
}
```

**Response**

```json
{
  "mensaje": "Análisis completado: Resumir en una oración",
  "modelo_ejecucion": "Llama-3.1-8b (Groq)",
  "respuesta": {
    "resultado": "La IA estudia cómo crear sistemas que aprenden."
  },
  "metricas": {
    "tiempo_respuesta_ms": 320
  }
}
```

## Estructura del proyecto

```
IA_GeckOS/
├── main.py            # Aplicación FastAPI y lógica de los endpoints
├── test_main.py       # Pruebas unitarias (pytest)
├── locustfile.py      # Pruebas de carga (Locust)
├── listar_modelos.py  # Utilidad para listar modelos disponibles en Google GenAI
├── requirements.txt   # Dependencias Python
└── .env               # Variables de entorno (no versionado)
```

## Pruebas

**Unitarias** (requiere `pytest`):

```bash
pip install pytest
pytest test_main.py -v
```

**Carga** (requiere `locust`):

```bash
pip install locust
locust -f locustfile.py --host=http://localhost:8000
```

El archivo `locustfile.py` simula patrones de uso de un estudiante con `test_mode: true` para evitar consumo de APIs en entornos de prueba.

## Stack tecnológico

- **Framework:** FastAPI
- **Servidor ASGI:** Uvicorn
- **Validación:** Pydantic
- **LLM / Embeddings:** Google GenAI SDK, Groq, Cohere
- **Generación de imágenes:** Gradio Client (Hugging Face Spaces)
- **Búsqueda léxica:** rank-bm25
- **Configuración:** python-dotenv

