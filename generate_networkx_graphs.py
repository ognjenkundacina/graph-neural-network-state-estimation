import networkx as nx
from networkx.readwrite import json_graph
import json
import math
import os
import category_encoders as ce
import pandas as pd
from csv import *


def generate_datasets():
    data_dir = os.path.abspath("./data_from_wls_se_solver")

    generate_dataset(datasetType="train", data_dir=data_dir, should_connect_node_to_second_neighbours=True)

    generate_dataset(datasetType="test", data_dir=data_dir, should_connect_node_to_second_neighbours=True)

    generate_dataset(datasetType="validation", data_dir=data_dir, should_connect_node_to_second_neighbours=True)


def generate_dataset(datasetType, data_dir, should_connect_node_to_second_neighbours=True):
    """

    :param datasetType: train, validation or test set
    :param should_connect_node_to_second_neighbours: if true, adds direct variable-to-variable node connections, i.e.
           creates the augmented factor graph from the factor graph

    local variables:
    numMeasurements - size of measurement vector, number of factor nodes, it is two times larger than the number of measurement phasors
    numGraphs - number of samples in the dataset
    """
    print("reading from wls se files started")
    numGraphs, numVariableNodes, numMeasurements, estimate_rows, jacobian_rows, measurement_rows, variance_rows, covariance_rows = read_from_wls_se_files(
        datasetType, data_dir)
    print("reading from wls se files done")

    jsonFilePath = open_json_dataset_file(datasetType)

    encoded_variable_node_indices = encode_variable_node_indices(numVariableNodes)

    jacobRowCount = 0
    for iGraph in range(numGraphs):
        if iGraph % 10 == 0:
            print(iGraph)

        G = nx.DiGraph()  # undirected graph

        measurement_row = measurement_rows[iGraph]
        variance_row = variance_rows[iGraph]
        estimate_row = estimate_rows[iGraph]
        covariance_row_temp = covariance_rows[iGraph]
        # since the number of covariances is two times smaller than the number of factor nodes, because of easier implementation
        # we double the number of covariances in the covariance list.
        covariance_row = []
        for x in covariance_row_temp:
            covariance_row.append(x)
            covariance_row.append(x)

        if len(measurement_row) != numMeasurements:
            print("ERROR: len(measurement_row) != numMeasurements")
            return
        if len(variance_row) != numMeasurements:
            print("ERROR: len(variance_row) != numMeasurements")
            return
        if len(covariance_row) != numMeasurements:
            print("ERROR: len(covariance_row) != numMeasurements")
            return
        if len(estimate_row) != numVariableNodes:
            print("ERROR: len(estimate_row) != numVariableNodes")
            return

        add_variable_nodes(G, estimate_row, encoded_variable_node_indices, numVariableNodes, iGraph)

        add_factor_nodes(G, covariance_row, measurement_row, numMeasurements, numVariableNodes, variance_row)

        if G.number_of_nodes() != numMeasurements + numVariableNodes:
            print("ERROR: G.number_of_nodes() != numMeasurements + numVariableNodes")
            return

        jacobRowCount = add_graph_edges(G, jacobRowCount, jacobian_rows, numMeasurements, numVariableNodes)

        connect_nodes_to_second_neighbours(G, numVariableNodes, should_connect_node_to_second_neighbours)

        parced_graph = json_graph.node_link_data(G)
        with open(jsonFilePath, 'a') as f:
            json.dump(parced_graph, f)
            f.write(",")

    with open(jsonFilePath, mode="r+") as file:
        file.seek(os.stat(jsonFilePath).st_size - 1)  # override the last comma in the file
        file.write("]")


def encode_variable_node_indices(numVariableNodes):
    data = pd.DataFrame(
        {'nodeIdx': [i for i in range(numVariableNodes)]})
    encoder = ce.BaseNEncoder(cols=['nodeIdx'], return_df=False, base=2)
    encoded_variable_node_indices = encoder.fit_transform(data)
    return encoded_variable_node_indices


def open_json_dataset_file(datasetType):
    if datasetType == "test":
        with open(os.path.join('data/test', 'data.json'), 'w') as json_file:
            json_file.write("[")
            jsonFilePath = os.path.join('data/test', 'data.json')
    elif datasetType == "validation":
        with open(os.path.join('data/validation', 'data.json'), 'w') as json_file:
            json_file.write("[")
            jsonFilePath = os.path.join('data/validation', 'data.json')
    else:
        jsonFilePath = os.path.join('data/train', 'data.json')
        with open(os.path.join('data/train', 'data.json'), 'w') as json_file:
            json_file.write("[")
    return jsonFilePath


def add_graph_edges(G, jacobRowCount, jacobian_rows, numMeasurements, numVariableNodes):
    # jacobRowCount = 0  # sluzi da bismo prolazili po svim grafovima u okviru jacobian matrice
    # variable nodes: 0..numVariables-1
    # factor nodes: numVariables..numVariables + numMeasurements - 1
    for iMeasurement in range(numMeasurements):
        jacobRow = jacobian_rows[jacobRowCount]
        jacobRowCount += 1

        factorNodeIndex = iMeasurement + numVariableNodes
        for iVariable in range(numVariableNodes):
            if abs(float(jacobRow[iVariable])) > 0.0001:
                G.add_edge(str(factorNodeIndex), str(iVariable))
                # IGNNITION received as input an undirected graph, even though it only
                # supports (at the moment) directed graphs -> therefore we must double the number of edges.
                G.add_edge(str(iVariable), str(factorNodeIndex))
    return jacobRowCount


def add_variable_nodes(G, estimate_row, encoded_variable_node_indices, numVariableNodes, iGraph):
    for iVar in range(numVariableNodes):
        index_encoding = encoded_variable_node_indices[iVar].tolist()
        #G.add_node(str(iVar), entity='variableNode', voltage=estimate_row[iVar], index_encoding=index_encoding)
        G.add_node(str(iVar), entity='variableNode', iGraph=iGraph, index_encoding=index_encoding)
        # self loop:
        G.add_edge(str(iVar), str(iVar))


def add_factor_nodes(G, covariance_row, measurement_row, numMeasurements, numVariableNodes, variance_row):
    for iMeasur in range(numMeasurements):
        meas = float(measurement_row[iMeasur])
        var = math.log10(float(variance_row[iMeasur])) / 10.0
        covar = float(covariance_row[iMeasur]) * 10.0
        factorNodeIndex = iMeasur + numVariableNodes
        G.add_node(str(factorNodeIndex), entity='factorNode', measurement=meas, variance=var, covariance=covar)
        # self loop:
        G.add_edge(str(factorNodeIndex), str(factorNodeIndex))


def connect_nodes_to_second_neighbours(G, numVariableNodes, should_connect_node_to_second_neighbours):
    if should_connect_node_to_second_neighbours:
        for iVariable in range(numVariableNodes):
            connect_node_to_second_neighbours(G, str(iVariable))


def get_second_neighbors(G, node):
    return [nodeId for nodeId, pathLength in nx.single_source_shortest_path_length(G, node, cutoff=2).items() if
            pathLength == 2]


def connect_node_to_second_neighbours(G, variableNodeId):
    for neighrbVariableNodeId in get_second_neighbors(G, variableNodeId):
        G.add_edge(variableNodeId, neighrbVariableNodeId)


def read_from_wls_se_files(datasetType, data_dir):
    if datasetType == "test":
        path = str(data_dir) + "/Test_Estimate.csv"
    elif datasetType == "validation":
        path = str(data_dir) + "/Validation_Estimate.csv"
    else:
        path = str(data_dir) + "/Training_Estimate.csv"
    with open(path, 'r') as read_obj:
        csv_reader = reader(read_obj)
        estimate_rows = list(csv_reader)

    if datasetType == "test":
        path = str(data_dir) + "/Test_Jacobian.csv"
    elif datasetType == "validation":
        path = str(data_dir) + "/Validation_Jacobian.csv"
    else:
        path = str(data_dir) + "/Training_Jacobian.csv"
    with open(path, 'r') as read_obj:
        csv_reader = reader(read_obj)
        jacobian_rows = []
        for i, line in enumerate(csv_reader):
            jacobian_row = [1 if abs(float(element)) > 0.0001 else 0 for element in line]
            jacobian_rows.append(jacobian_row)

    if datasetType == "test":
        path = str(data_dir) + "/Test_Measurement.csv"
    elif datasetType == "validation":
        path = str(data_dir) + "/Validation_Measurement.csv"
    else:
        path = str(data_dir) + "/Training_Measurement.csv"
    with open(path, 'r') as read_obj:
        csv_reader = reader(read_obj)
        measurement_rows = list(csv_reader)

    if datasetType == "test":
        path = str(data_dir) + "/Test_Variance.csv"
    elif datasetType == "validation":
        path = str(data_dir) + "/Validation_Variance.csv"
    else:
        path = str(data_dir) + "/Training_Variance.csv"
    with open(path, 'r') as read_obj:
        csv_reader = reader(read_obj)
        variance_rows = list(csv_reader)

    if datasetType == "test":
        path = str(data_dir) + "/Test_Covariance.csv"
    elif datasetType == "validation":
        path = str(data_dir) + "/Validation_Covariance.csv"
    else:
        path = str(data_dir) + "/Training_Covariance.csv"
    with open(path, 'r') as read_obj:
        csv_reader = reader(read_obj)
        covariance_rows = list(csv_reader)

    if datasetType == "test":
        path = str(data_dir) + "/Test_NumMeasurements.txt"
    elif datasetType == "validation":
        path = str(data_dir) + "/Validation_NumMeasurements.txt"
    else:
        path = str(data_dir) + "/Training_NumMeasurements.txt"
    file1 = open(path, 'r')
    lines = file1.readlines()
    numLinePMeasurements = int(lines[0])
    numPInjMeasurements = int(lines[1])
    numThetaMeasurements = int(lines[2])

    numGraphs = int(lines[3])
    numVariableNodes = int(lines[4]) * 2  # Vream and Vimag
    numMeasurements = numLinePMeasurements + numPInjMeasurements + numThetaMeasurements

    if len(jacobian_rows) != numGraphs * numMeasurements:
        print(
            "ERROR: len(jacobian_rows) != numGraphs * (numLinePMeasurements + numPInjMeasurements + numThetaMeasurements)")
        print("len(jacobian_rows) ", len(jacobian_rows))
        print("numGraphs ", numGraphs)
        print("numLinePMeasurements + numPInjMeasurements + numThetaMeasurements  ",
              numLinePMeasurements + numPInjMeasurements + numThetaMeasurements)
        return

    if len(jacobian_rows[0]) != numVariableNodes:
        print("ERROR: len(jacobian_rows[0]) != numVariableNodes")
        return

    if len(estimate_rows) != numGraphs:
        print("ERROR: len(estimate_rows) != numGraphs")
        return

    if len(estimate_rows[0]) != numVariableNodes:
        print("ERROR: len(estimate_rows[0]) != numVariableNodes")
        return

    if len(measurement_rows) != numGraphs:
        print("ERROR: len(measurement_rows) != numGraphs")
        return

    if len(measurement_rows[0]) != numMeasurements:
        print("ERROR: len(measurement_rows[0]) != numMeasurements")
        return

    if len(variance_rows) != numGraphs:
        print("ERROR: len(variance_rows) != numGraphs")
        return

    if len(variance_rows[0]) != numMeasurements:
        print("ERROR: len(variance_rows[0]) != numMeasurements")
        return

    if len(covariance_rows) != numGraphs:
        print("ERROR: len(covariance_rows) != numGraphs")
        return

    # there is one covariance value per measurement phasor, while numMeasurements is the size of
    # measurement vector, i.e. it is two times larger than the number of measurement phasors
    if 2 * len(covariance_rows[0]) != numMeasurements:
        print("ERROR: len(variance_rows[0]) != numMeasurements")
        return

    return numGraphs, numVariableNodes, numMeasurements, estimate_rows, jacobian_rows, measurement_rows, variance_rows, covariance_rows


generate_datasets()
