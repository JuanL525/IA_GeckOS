from locust import HttpUser, task, between

class EstudianteGeckOS(HttpUser):
    # El estudiante piensa entre 1 y 3 segundos antes de hacer otra acción
    wait_time = between(1.0, 3.0)

    @task(4) # Alta probabilidad: El estudiante usa mucho el chat
    def probar_chat(self):
        payload = {
            "mensaje": "¿Me explicas qué es un framework?",
            "test_mode": True  
        }
        with self.client.post("/chat", json=payload, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Falló chat: {response.status_code}")

    @task(2) # Probabilidad media: Búsqueda de archivos
    def probar_busqueda(self):
        payload = {
            "consulta": "universidad",
            "archivos": [
                {"id": "1", "nombre": "clases.txt", "contenido": "Tengo que ir a la facultad de ingeniería."},
                {"id": "2", "nombre": "gastos.txt", "contenido": "Comprar pan y leche."}
            ],
            "test_mode": True
        }
        with self.client.post("/buscar", json=payload, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Falló búsqueda: {response.status_code}")

    @task(2) # Probabilidad media: Análisis de textos
    def probar_analisis(self):
        payload = {
            "texto": "La Inteligencia Artificial es el campo de estudio que busca crear sistemas capaces de aprender.",
            "accion": "Traducir a inglés",
            "test_mode": True 
        }
        with self.client.post("/analizar-documento", json=payload, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Falló análisis: {response.status_code}")
