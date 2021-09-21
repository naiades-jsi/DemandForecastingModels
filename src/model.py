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
        # TODO: here the model recieves a dictionarry {"timestamp": ..., "feature_vector": ...}, makes predictions and sends out an output to kafka
        prediction = self.nn.predict(message_value["feature_vector"])
        
    def build_train_model(self, model_structure: Dict[str, Any], train_file: str):
        self.nn = Sequential()
        self.nn.add(Masking(mask_value=0., input_shape=(model_structure["n_of_timesteps"], model_structure["num_features"])))
        self.nn.add(LSTM(1, activation = 'tanh', input_shape = (model_structure["n_of_timesteps"], model_structure["num_features"]), return_sequences=True))
        self.nn.add(Dropout(model_structure["dropout"]))
        self.nn.add(LSTM(NCells))
        self.nn.add(Dropout(model_structure["dropout"]))
        self.nn.add(Dense(1))
        self.nn.compile(loss = 'mse', optimizer='adam')
        X_ = ma.filled(train_X_data[i],0)
        Y_ = ma.filled(train_Y_data[i],0)
        MODEL = model.fit(X_, Y_, epochs = model_structure["epochs"], batch_size = model_structure["batch_size"],
                        validation_split = modelstructure["validation_split"], shuffle = False,
                        callbacks=[earlystopping])