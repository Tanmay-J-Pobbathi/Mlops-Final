from locust import HttpUser, task , between#between was added for it to work
class MLTest(HttpUser):
     wait_time = between(1, 3) #this line to define class and give a wait time
     @task 
     def predict(self): 
         self.client.post("/predict", json={"features": [5.1, 3.5, 1.4, 0.2]}) 