from typing import Any, Dict, List

class Model():
    training_data: str
    model_structure: Dict[str, Any]

    def __init__(self, conf: Dict[Any, Any] = None) -> None:
        super().__init__()
        if(conf is not None):
            self.configure(conf)

    def configure(self, conf: Dict[Any, Any] = None,
                  configuration_location: str = None,
                  algorithm_indx: int = None) -> None:
        self.training_data = conf["training_data"]
        self.model_structure = conf["model_structure"]

        # Build and train the model
        self.build_train_model(model_structure=self.model_structure,
                               train_file=self.training_data)

    def message_insert(self, message_value: Dict[str, Any]) -> Any:
        pass
        # TODO: here the model recieves a dictionarry {"timestamp": ..., "feature_vector": ...}, makes predictions and sends out an output to kafka

    def build_train_model(self, model_structure: Dict[str, Any], train_file: str):
        pass
        # TODO: the code to build and train the model 
