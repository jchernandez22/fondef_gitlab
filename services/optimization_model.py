import gurobipy as gp 
from gurobipy import GRB
from gurobi_ml import add_predictor_constr
import pandas as pd
import time
from tqdm import tqdm
import random
import numpy as np

class OptModel():

    def __init__(self, n_scen, model):
        self.K_scen = n_scen
        self.model = model

    def optimize(self, nn_model):
        
        start = time.time()
        m = gp.Model("Portafolio de Mitigación de una Red de Salud")
        K = self.K_scen
        H = df_hospitales_sin_refuerzo.shape[0] # numero de hospitales

        # Conjuntos y Parámetros
        S = df_hospitales_sin_refuerzo.shape[1] 
        x = m.addVars(H, name="x", vtype=GRB.BINARY)
        y = m.addVars(K, 3, name="y", vtype=GRB.BINARY)
        F = m.addVars(K, H, S, name= "F", vtype=GRB.INTEGER)
        tiempo_pred = m.addVars(K ,name="costo_predecido", vtype = GRB.CONTINUOUS) ###### CUIDADO ######

        m.addConstr(sum(x[h] for h in range(H)) == 1)
        m.addConstrs(sum(y[k, i] for i in range(3)) == 1 for k in range(K))
        m.addConstrs(F[k, h, s] == df_hospitales_sin_refuerzo.iat[h, s]*(1-x[h]) + df_hospitales_con_refuerzo.iat[h,s]*x[h] for k in range(K) for h in range(H) for s in range(S))
        m.update()

        # Input
        id_K_escenarios = random.sample(escenarios, K)

        input = pd.DataFrame(columns=X.columns)
        rows = []
        for id_scenarios, id_scen_terremoto, decision in id_K_escenarios:
            rows.append(X[(X['id_escen']==int(id_scenarios)) & (X['id_escen_terremoto']==int(id_scen_terremoto)) & (X[f'y_{decision}']==1)].copy()) #store the DataFrames to a dict of dict
        input = pd.concat(rows)
        input = input.iloc[:,2:].copy()

        for k in range(K):
            for i in range(3):
                input.iat[k, i] = y[k, i]
            
            for s in range(S):
                for h in range(H):
                    input.iat[k, 3 + 152*4 + S*h + s] = F[k, h, s]
                
        for k in range(K):
            pred_constr = add_predictor_constr(
                m, nn_model, pd.DataFrame(input.iloc[k,:]).T, tiempo_pred[k]) 
                
        m.setObjective((1/K) * sum(tiempo_pred[k] for k in range(K)))
        m.ModelSense = GRB.MINIMIZE

        m.Params.OutputFlag = 0
        m.update()
        m.optimize() 
        opt_value = m.objVal
        x_values = np.array([v.X for v in x.values()])
        end=time.time()

        return opt_value, x_values, (end - start)
        