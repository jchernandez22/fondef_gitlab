import time
from services.optimization_model import OptModel
from services.nn_model import NNModel

if __name__ == '__main__':
        
    params = {
        "K_escen": 10,
        "escen_mermas_terremoto": 1,
        "n_hospitales": 4,
        "hospitales_campaña_activos" : ["Ubicacion 1", "Ubicacion 3"],
        "train_model": True,       # False si se quiere cargar modelo
        "model_layers": [4]*4,
        "model_activation_function": "relu" # "identity"
    }

    nn_model = NNModel(activation=params['model_activation_function'],
                       layers=params['model_layers'],
                       train=params['train_model'])
        
    opt_model = OptModel(params["K_escen"], nn_model)
 
    
