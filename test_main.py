import os
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, mock_open

# IMPORTANTE: Cambia 'main' por el nombre real de tu archivo de FastAPI (sin el .py)
from main import app 

# Creamos un cliente falso de FastAPI para simular las peticiones
client = TestClient(app)

# Configuramos variables de entorno falsas para que tu validación no falle
os.environ["GOOGLE_API_KEY"] = "fake_google_key"
os.environ["HUGGINGFACE_API_KEY"] = "fake_hf_key"
os.environ["GROQ_API_KEY"] = "fake_groq_key"


# ==========================================
# PRUEBAS PARA EL ENDPOINT /chat
# ==========================================

@patch("main.genai.Client")
def test_chat_respuesta_exitosa(mock_genai_client):
    mock_instance = MagicMock()
    mock_response = MagicMock()
    mock_response.text = '{"mensaje": "Esta es una respuesta simulada por pytest"}' 
    
    mock_instance.models.generate_content.return_value = mock_response
    mock_genai_client.return_value = mock_instance

    response = client.post("/chat", json={"mensaje": "¿Qué es un framework?"})

    assert response.status_code == 200
    data = response.json()
    assert "respuesta" in data
    assert data["respuesta"]["mensaje"] == "Esta es una respuesta simulada por pytest"
    assert "metricas" in data
    assert "tiempo_respuesta_ms" in data["metricas"]


# ==========================================
# PRUEBAS PARA EL ENDPOINT /generar-fondo
# ==========================================

@patch("main.Client") 
@patch("builtins.open", new_callable=mock_open, read_data=b"datos_falsos_de_imagen")
def test_generar_fondo_plan_a_exitoso(mock_file, mock_gradio_client):
    mock_instance = MagicMock()
    mock_instance.predict.return_value = ["/ruta/falsa/imagen_generada.webp"]
    mock_gradio_client.return_value = mock_instance

    response = client.post("/generar-fondo", json={"descripcion": "Un paisaje de Quito"})

    assert response.status_code == 200
    data = response.json()
    
    assert "FLUX.1-dev" in data["mensaje"]
    
    assert data["imagen"].startswith("data:image/webp;base64,")
    
    mock_file.assert_called_with("/ruta/falsa/imagen_generada.webp", "rb")


@patch("main.Client")
@patch("builtins.open", new_callable=mock_open, read_data=b"datos_falsos_de_imagen")
def test_generar_fondo_fallback_plan_b(mock_file, mock_gradio_client):

    mock_flux_instance = MagicMock()
    mock_flux_instance.predict.side_effect = Exception("Servidor FLUX caído")
    
    mock_ernie_instance = MagicMock()
    mock_ernie_instance.predict.return_value = ["/ruta/falsa/imagen_ernie.webp"]
    
    mock_gradio_client.side_effect = [mock_flux_instance, mock_ernie_instance]

    response = client.post("/generar-fondo", json={"descripcion": "Un test de fallback"})

    assert response.status_code == 200
    data = response.json()
    
    assert "ERNIE-Image-Turbo" in data["mensaje"]

# ==========================================
# PRUEBAS PARA EL ENDPOINT /buscar
# ==========================================

@patch("main.es_contenido_valido", return_value=True) # <-- Engañamos a tu nuevo filtro
@patch("main.genai.Client")
def test_buscar_plan_a_gemini(mock_genai_client, mock_filtro):
    # 1. PREPARAR EL MOCK DE GEMINI
    mock_instance = MagicMock()
    mock_embed_response = MagicMock()

    # Vectores matemáticos estáticos
    mock_emb_consulta = MagicMock()
    mock_emb_consulta.values = [1.0, 0.0, 0.0] 
    
    mock_emb_archivo1 = MagicMock()
    mock_emb_archivo1.values = [1.0, 0.0, 0.0] # 100% de similitud semántica
    
    mock_emb_archivo2 = MagicMock()
    mock_emb_archivo2.values = [0.0, 1.0, 0.0] # 0% de similitud semántica

    mock_embed_response.embeddings = [mock_emb_consulta, mock_emb_archivo1, mock_emb_archivo2]
    mock_instance.models.embed_content.return_value = mock_embed_response
    mock_genai_client.return_value = mock_instance

    # 2. EJECUTAR LA PRUEBA
    payload = {
        "consulta": "universidad",
        "archivos": [
            {"id": "1", "nombre": "clases.txt", "contenido": "Texto válido sobre la universidad"},
            {"id": "2", "nombre": "basura.txt", "contenido": "Texto irrelevante"}
        ]
    }
    response = client.post("/buscar", json=payload)

    # 3. VERIFICAR RESULTADOS
    assert response.status_code == 200
    data = response.json()
    assert "Gemini" in data["mensaje"]
    assert len(data["resultados"]) == 1 # El archivo de 0% debió ser eliminado
    assert data["resultados"][0]["id"] == "1"


@patch("main.es_contenido_valido", return_value=True)
@patch("main.cohere.Client")
@patch("main.genai.Client")
def test_buscar_fallback_cohere(mock_genai_client, mock_cohere_client, mock_filtro):
    # 1. FORZAMOS EL FALLO EN GEMINI
    mock_genai_client.side_effect = Exception("Google colapsado")

    # 2. PREPARAMOS EL MOCK DE COHERE
    mock_cohere_instance = MagicMock()
    mock_resp_consulta = MagicMock()
    mock_resp_consulta.embeddings = [[1.0, 0.0, 0.0]]
    
    mock_resp_archivos = MagicMock()
    mock_resp_archivos.embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]

    # Cohere usa 2 llamadas (query y document)
    mock_cohere_instance.embed.side_effect = [mock_resp_consulta, mock_resp_archivos]
    mock_cohere_client.return_value = mock_cohere_instance

    # 3. EJECUTAR LA PRUEBA
    payload = {
        "consulta": "ingeniería",
        "archivos": [
            {"id": "1", "nombre": "clases.txt", "contenido": "Facultad de ingeniería"},
            {"id": "2", "nombre": "random.txt", "contenido": "Nada que ver"}
        ]
    }
    response = client.post("/buscar", json=payload)

    # 4. VERIFICAR RESULTADOS
    assert response.status_code == 200
    data = response.json()
    assert "Cohere" in data["mensaje"] # Validamos que el Fallback nos salvó la vida


@patch("main.es_contenido_valido", return_value=False) # <-- Simulamos que todo es basura
def test_buscar_filtro_basura(mock_filtro):
    payload = {
        "consulta": "universidad",
        "archivos": [
            {"id": "1", "nombre": "asdasd.txt", "contenido": "asdasdasdasd"}
        ]
    }
    response = client.post("/buscar", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert "No se encontraron archivos con contenido legible" in data["mensaje"]
    assert len(data["resultados"]) == 0

    # ==========================================
# PRUEBAS PARA EL ENDPOINT /analizar-documento
# ==========================================

@patch("main.Groq")
@patch("main.genai.Client")
def test_analizar_tarea_general_groq(mock_gemini, mock_groq):
    # 1. PREPARAMOS EL MOCK DE GROQ
    mock_groq_instance = MagicMock()
    mock_choice = MagicMock()
    # Simulamos que Llama 3 devuelve un JSON válido
    mock_choice.message.content = '{"resultado": "Este es un resumen super rápido hecho por Groq"}'
    
    mock_response_groq = MagicMock()
    mock_response_groq.choices = [mock_choice]
    
    mock_groq_instance.chat.completions.create.return_value = mock_response_groq
    mock_groq.return_value = mock_groq_instance

    # 2. EJECUTAR PRUEBA (Acción normal)
    payload = {"texto": "Había una vez un perro.", "accion": "Resumir en 2 palabras"}
    response = client.post("/analizar-documento", json=payload)

    # 3. VERIFICAR RESULTADOS
    assert response.status_code == 200
    data = response.json()
    assert "Groq" in data["modelo_ejecucion"]
    assert data["respuesta"]["resultado"] == "Este es un resumen super rápido hecho por Groq"
    # Verificamos que Gemini NUNCA fue llamado para esta tarea
    mock_gemini.assert_not_called()


@patch("main.Groq")
@patch("main.genai.Client")
def test_analizar_traduccion_enrutada_a_gemini(mock_gemini, mock_groq):
    # 1. PREPARAMOS EL MOCK DE GEMINI
    mock_gemini_instance = MagicMock()
    mock_response_gemini = MagicMock()
    mock_response_gemini.text = '{"resultado": "Once upon a time there was a dog."}'
    
    mock_gemini_instance.models.generate_content.return_value = mock_response_gemini
    mock_gemini.return_value = mock_gemini_instance

    # 2. EJECUTAR PRUEBA (Acción con palabra clave 'traducir')
    payload = {"texto": "Había una vez un perro.", "accion": "Traducir esto al inglés"}
    response = client.post("/analizar-documento", json=payload)

    # 3. VERIFICAR RESULTADOS
    assert response.status_code == 200
    data = response.json()
    assert "Traductor Dedicado" in data["modelo_ejecucion"]
    assert data["respuesta"]["resultado"] == "Once upon a time there was a dog."
    # Verificamos que el Enrutador fue inteligente y se saltó Groq
    mock_groq.assert_not_called()


@patch("main.Groq")
@patch("main.genai.Client")
def test_analizar_fallback_rescate_gemini(mock_gemini, mock_groq):
    # 1. FORZAMOS EL FALLO EN GROQ
    mock_groq_instance = MagicMock()
    mock_groq_instance.chat.completions.create.side_effect = Exception("Groq está colapsado")
    mock_groq.return_value = mock_groq_instance

    # 2. PREPARAMOS EL MOCK DE GEMINI PARA EL RESCATE
    mock_gemini_instance = MagicMock()
    mock_response_gemini = MagicMock()
    mock_response_gemini.text = '{"resultado": "Resumen rescatado por Gemini"}'
    
    mock_gemini_instance.models.generate_content.return_value = mock_response_gemini
    mock_gemini.return_value = mock_gemini_instance

    # 3. EJECUTAR PRUEBA (Acción normal, pero Groq fallará)
    payload = {"texto": "Texto de emergencia", "accion": "Resumir"}
    response = client.post("/analizar-documento", json=payload)

    # 4. VERIFICAR RESULTADOS
    assert response.status_code == 200
    data = response.json()
    assert "Fallback" in data["modelo_ejecucion"]
    assert data["respuesta"]["resultado"] == "Resumen rescatado por Gemini"