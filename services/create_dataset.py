from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import json
import os

def preprocesamiento_SSO_casualties(id_escenario, path_data):
    """
    Carga el dataset de heridos, revisa agmodela los clusters faltantes y conserva las columnas útiles.
    """
    data = pd.read_csv(
        path_data + f'SSO_casualties_clustered/scen{id_escenario}_casualties_by_manz_clustered.csv').loc[:, ['COD_NVO','S1', 'S2', 'S3', 'S4']]
    data = data.groupby('COD_NVO', as_index = False).sum()
    return data.loc[:, ['S1', 'S2', 'S3', 'S4']]

def preprocesamiento_capacidades_terremoto(id_escenario_terremoto, path_data):
    servidores_hospitales = ['Sala Categorizacion', 'Reanimador', 'Box Atencion', 'Box Trauma',
       'Berger', 'Sala RX', 'Box Observacion Critico',
       'Box Observacion No Critico', 'Box C4C5']
    servidores_sapu_sar = ['Sala Categorizacion', 'Box Reanimacion', 'Box Atencion',
       'Box Observacion', 'Sala RX', 'Sala Tratamientos']
    
    capacidades_hospitales = pd.read_csv(
        path_data + f'capacidades_terremoto/capacidades_hospitales_terremoto_scen{id_escenario_terremoto}.csv', sep=";")
    
    capacidades_sapu_sar = pd.read_csv(
        path_data + f'capacidades_terremoto/capacidades_sapu_sar_terremoto_scen{id_escenario_terremoto}.csv', sep=";")

    return capacidades_hospitales[servidores_hospitales], capacidades_sapu_sar[servidores_sapu_sar]

def concat_rows(df):
    new_cols = [
        f"{col}_{i}"
        for i in range(1, len(df)+1)
        for col in df.columns
    ]
    n_cols = len(df.columns)
    new_df = pd.DataFrame(
        df.values.reshape([-1, n_cols*len(df)]),
        columns=new_cols
    )
    return new_df

def preprocesamiento_input(id_escenario, id_escenario_terremoto, decision, path_data):
    """
    Función encargada de procesar los dataframes. En particular, carga los datos, realiza cambios
    que sean necesarios y tranforma los datasets en un vector unidimensional.
    """
    heridos = preprocesamiento_SSO_casualties(id_escenario, path_data)
    capacidades_hospitales, capacidades_sapu_sar = preprocesamiento_capacidades_terremoto(id_escenario_terremoto, path_data)

    df_id = pd.DataFrame([[int(id_escenario),int(id_escenario_terremoto),]], columns=["id_escen", "id_escen_terremoto"])

    decisions = np.array([[0,0,0]])
    decisions[0, decision] = 1

    df_decisions = pd.DataFrame(decisions, columns=['y_0', 'y_1', 'y_2'])
    heridos_concat = concat_rows(heridos)
    capacidades_hospitales_concat, capacidades_sapu_sar_concat = concat_rows(capacidades_hospitales), concat_rows(capacidades_sapu_sar)

    return pd.concat([df_id, df_decisions, heridos_concat, capacidades_hospitales_concat, capacidades_sapu_sar_concat], axis=1)

def generar_output(id_escenario, id_escenario_terremoto, decision, path_data, mean=True):

    hospitales = ['Hospital Sótero del río', 'Hospital San josé de maipo', 'Hospital Padre huratdo',
                'Hospital La florida']
    
    sapu_sar = ['SAPU Bernardo Leighton', 'SAPU Karol Wojtyla', 'SAPU La Granja', 
                'SAPU Padre Manuel Villaseca', 'SAR Santiago de Nueva Extremadura']

    file = open(path_data + f"metricas_simulaciones/metricas_escenario_{id_escenario}_{id_escenario_terremoto}_{decision}.json", "r")
    diccionario = json.load(file)

    tiempos_promedio = []
    muertos = []

    # hospitales
    for hospital in hospitales:
        tiempos_promedio.append(
            diccionario[hospital]['Tiempo de Espera Promedio'])
        muertos.append(
            diccionario[hospital]['Muertos'])

    # sapu_sar
    for establecimiento in sapu_sar:
        tiempos_promedio.append(
            diccionario[establecimiento]['Tiempo de Espera Promedio'])
        muertos.append(
            diccionario[establecimiento]['Muertos'])

    if mean:
        return np.array([np.array(np.mean(tiempos_promedio)),
                                np.array(np.sum(muertos))])
    else:
        return np.concatenate([np.array(tiempos_promedio), np.array(muertos)])

def create_dataset():
    data_folder_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/')

    escenarios = [j.split(".")[0].split("_")[2:] for j in [i for i in os.listdir(data_folder_path + f"metricas_simulaciones") if i.split("_")[0] == "metricas"]]

    X = []
    Y = []
    for escenario in escenarios:
        id_escenario_heridos = escenario[0]
        id_escenario_terremoto = escenario[1]
        decision = int(escenario[2])

        X.append(preprocesamiento_input(id_escenario_heridos, id_escenario_terremoto, decision, path_data=data_folder_path))
        Y.append(generar_output(id_escenario_heridos, id_escenario_terremoto, decision = decision, path_data=data_folder_path))

    X = pd.concat(X, axis=0)
    Y = np.vstack(Y)
    X["tiempo_promedio"], X["muertos"] = Y[:,0], Y[:,1]

    X.to_csv(data_folder_path + f"dataset/dataset.csv")

if __name__ == '__main__':

    create_dataset()
    #data_folder_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/')
    #print(preprocesamiento_SSO_casualties(2, data_folder_path))