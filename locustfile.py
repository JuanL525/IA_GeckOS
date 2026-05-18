from locust import HttpUser, task, between

class EstudianteGeckOS(HttpUser):
    # El estudiante piensa entre 1 y 3 segundos antes de hacer otra acción
    wait_time = between(1.0, 3.0)

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
