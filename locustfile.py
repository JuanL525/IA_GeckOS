from locust import HttpUser, task, between

class EstudianteGeckOS(HttpUser):
    wait_time = between(1.0, 3.0)

    @task(1) 
    def probar_generador_imagenes(self):
        payload = {
            "descripcion": "Un paisaje ciberpunk de Quito en el año 2077",
            "test_mode": True  
        }
        with self.client.post("/generar-fondo", json=payload, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Falló la imagen con estado {response.status_code}")
