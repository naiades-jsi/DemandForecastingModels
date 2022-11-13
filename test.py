# test_module.py

import unittest
from src.model import GDB_model

class TestModule(unittest.TestCase):
    def test_gdb(self):
        config = {
            "n_days": 7,
            "predicted_timesteps": 48,
            "n_features": 1,
            "fill_missing_data": "./data/MissingData/missingDataAlicanteUnialipark.npy",
            "max_missing_data_memory":500,
            "data": "./data/DataForModels/Univariate/data_alipark.csv",
            "model_name": "alicante_alipark",
            "model_file":"./LoadedModels/UnivariateGDB/alipark.pkl",
            "output": [],
            "output_conf": []
        }
        model = GDB_model(config)
        model.feature_vector_creation({"timestamp": 1668293402812, "ftr_vector": [30.0, 30.8, 27.0, 22.6, 22.2, 23.2, 24.0, 23.2, 20.8, 18.0, 20.4, 18.0, 16.8, 16.2, 17.0, 16.2, 18.8, 19.2, 16.4, 15.8, 17.4, 15.2, 12.2, 13.2]})

if __name__ == '__main__':
    unittest.main()


