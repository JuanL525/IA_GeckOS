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
