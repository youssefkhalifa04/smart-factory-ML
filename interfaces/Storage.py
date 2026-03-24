from abc import ABC, abstractmethod

class Storage(ABC):

    @abstractmethod
    def get_data(self, factory_id: str):
        pass
    @abstractmethod
    def push_notif(self, vector: list , factory_id: str , type: str , statement: str , code: str) -> bool:
        pass
    
    @abstractmethod
    def fake_data(self, factory_id: str): 
        pass
    @abstractmethod
    def last_trained_model(self, factory_id: str) -> str:
        pass
    @abstractmethod
    def get_latest_data(self, factory_id: str) -> str:
        pass
    @abstractmethod
    def push_prediction(self, factory_id: str, prediction: float) -> bool:
        pass
    
