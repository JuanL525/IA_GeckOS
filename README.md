# 🧠 Microservicio IA - MiDesk

Este repositorio contiene el microservicio de Inteligencia Artificial para **MiDesk**, un simulador de entorno de escritorio (Sistema Operativo Virtual). Está construido con **Python y FastAPI**, y actúa como el "cerebro" del sistema, permitiendo controlar la interfaz, realizar búsquedas inteligentes y personalizar el entorno visual.

## ✨ Características Principales

1. **🤖 Bot Agente Controlador:** Analiza las intenciones del usuario en lenguaje natural y devuelve comandos estructurados en JSON (ej. `abrir` aplicaciones, `crear_nota`) para que el frontend de MiDesk reaccione dinámicamente. Desarrollado con **Google Gemini (2.5 Flash)**.
2. **🔍 Búsqueda Semántica Inteligente:** Un buscador estilo "Spotlight" que encuentra archivos virtuales por su *significado* y contexto utilizando vectores y similitud del coseno. Desarrollado con **Gemini Embeddings (`gemini-embedding-001`)**.
3. **🎨 Generador de Fondos de Pantalla:** Creación de imágenes panorámicas (16:9) mediante texto para personalizar el escritorio de MiDesk. Impulsado por la API de **Hugging Face (Stable Diffusion XL)**.

---

## 🛠️ Tecnologías Utilizadas

* **Framework:** [FastAPI](https://fastapi.tiangolo.com/) (Python)
* **Servidor ASGI:** Uvicorn
* **Modelos de IA:** * Google GenAI SDK (Gemini 2.5 Flash / Embeddings)
  * Hugging Face Inference API (Stable Diffusion)
* **Gestión de Entorno:** `python-dotenv`

---

## 🚀 Instalación y Configuración Local

Sigue estos pasos para ejecutar el microservicio en tu máquina local:

### 1. Clonar el repositorio
```bash
git clone [https://github.com/TU_USUARIO/midesk-ia-backend.git](https://github.com/TU_USUARIO/midesk-ia-backend.git)
cd midesk-ia-backend
