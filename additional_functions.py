import tensorflow as tf
import os
from csv import reader
import numpy as np


"""
Taskovi:
[DONE] 1. Dodati iGraph u variable node inputs. Skloniti labele napona (estimates), da ne bismo slucajno koristili MSE za loss funkciju
[DONE ]2. Izgenerisi trening i test set sa ovim podacima
[DONE] 3. Doradi readout funkciju kao sto su ti rekli iz ignnition tima i prekonstrolisi sve ostale funkcije, kao i citav njihov repo 

TRENUTNO:
- srediti trenutno pucanje tako sto ces nakon poziva csv readera sve vrijednosti odmah konvertovati u float! 
makar isao element po element
- testirati da li se numStateVariables dobro izracuna tako sto cemo umjesto
return WRSS_GNN / numStateVariables
pisati:
return WRSS_GNN / 60
i vidjeti da li se slican training loss dobija nakon prv eepohe
- ukoliko vec ne mozemo printati razne vrijednosti na konzolu (npr vrijednosti vektora..) pokusati ih ispisivati 
u neki fajl, orvoren u okviru WRSSLoss funkcije

5. Vidi da li mozda ucitavanje measurement i variance rows, kao i one_jacobian da se realizuje na pocetku skripte, prije poziva funkcije
ili mozda nekako kao staticki clan, izguglati kako se to u pajtonu radi
samo da ne bismo iznova ucitavali iz fajlova pri prakom pozivu funkcije 
--> cini mi se da se to vec automatski radi, kao da se neko kesiranje radi, jer idu brzo epizode.. mada nisam siguran

 


batch_size: 1 - radimo stochastic gradient descent i obezbjedjujemo da samo jedan training sample ulazi u loss funkciju


U ovom primjeru se prvo radi pooling nad svim node embeddinzima u grafu, a onda se na vrijednost propusta kroz neuronsku mrezu
readout:
- type: pooling
  type_pooling: sum
  input: [a]
  output_name: pooled_a
- type: feed_forward
  input: [pooled_a]
  nn_name: readout_model
  output_label: my_label


  Nama je potrebno nesto malo drugacije: da prvo propustimo kroz neuronsku mrezu svaki node embedding pojedinacno, dobijemo readout_value skalare,
  tj predikcije;  pa da ih onda sve konkateniramo u jedan vektor predikcija:
  readout:
- type: neural_network
  input: [transmitter_receiver_pair]
  nn_name: readout
  output_name: readout_value
- type: concat
  axis: 1
  input: [readout_value, $wmmse_power]
  output_label: [$path_loss]

  ovako onda te vektore dobavljamo u loss funkciji:
  wmmse_power = tf.expand_dims(y_pred[:, -1], axis=0)
  readout_value = tf.expand_dims(y_pred[:, -2], axis=0)

  pitanje:
  kako mjerene vrijednosti, kao i varijanse iz dataseta dostaviti loss funkciji? One se ipak mijenjaju iz primjera u primjer?
  1. Da li readout nekako raditi i nad factor nodeovima, ne samo vaiable, da bismo pokupili sve potrebne podatke? Tada bismo morali indeksirati i factor nodeove, 
  pa da na osnovu izlaza iz readouta rekonstruisemo vektor mjerenja, sortirajuci pdoatke po indeksima!
  2. Da li mozda staviti za svaki podatak dodati input kom training sampleu u training setu pripada, pa da na osnovu toga izvucemo konkretan red iz 
  variance i measurement rows, umjesto da radimo komplikovan readout.
  (mozda cak i shuflle training set na false, da bi bila manja vremena pretrage odgovarajuceg reda u trening sampleu)
  - mislim da je druga opcija mnogo jednostavnija za implementirati. Dakle najbolje u svaki variable node staviti i ulazni podatak iGraph
 U readout funkciji umjesto $wmmse_power staviti $iGraph. 
 - Onda u loss funkciji uzeti iGraph samo od jedne predikcije i na osnovu njega dobaviti odgovarajuci red iz measurement rows i variance rows
  (trebalo bi da je ista vrijednost za iGraph za sve predikcije u vektoru -  ovo ce ujedno biti i dobar test radi li sve dobro, 
  tako sto cemo odstampati vrijednost citavog igraph vektora i vidjeti jesu li sve iste vrijednosti u njemu)

"""


"""
In the additional_functions.py file you may define a custom loss function (e.g., “sum_rate_loss” in the example).
 This function must have “y_true” and “y_pred” as input arguments, as in the example. 
 In this function, “y_true” are the training labels defined as “output_label” in the model_description.yaml file 
 (i.e., $path_loss in the example), while “y_pred” is a list that contains the tensors previously defined in the 
 "concat" function. In the example we define [readout_value, $wmmse_power] as "input" to the concat function 
 (model_description.yaml; line 37). Then, you can access the two features as follows:

"""
@tf.function()
def WRSSLoss(y_true, y_pred):

    N = tf.shape(y_pred)[0]
    #print("Velicina vektora state varijabli: ", tf.get_static_value(N))

    iGraphVector = tf.expand_dims(y_pred[:, -1], axis=0)
    voltageVector = tf.expand_dims(y_pred[:, -2], axis=0)

    #print(tf.get_static_value(iGraphVector))
    #print(tf.get_static_value(voltageVector))


    data_dir = os.path.abspath("./data_from_wls_se_solver")
    path = str(data_dir) + "/Test_Estimate.csv"
    with open(path, 'r') as read_obj:
        csv_reader = reader(read_obj)
        estimate_rows = list(csv_reader)


    path=str(data_dir) + "/Test_Measurement.csv"
    with open(path, 'r') as read_obj:
        csv_reader = reader(read_obj)
        measurement_rows = list(csv_reader)

    path = str(data_dir) + "/Test_Variance.csv"
    with open(path, 'r') as read_obj:
        csv_reader = reader(read_obj)
        variance_rows = list(csv_reader)

    path = str(data_dir) + "/one_jacobian.csv"
    csv_reader = reader(open(path, "rt"), delimiter=",")
    x = list(csv_reader)
    jacobian = np.array(x).astype("float")

    if (len(estimate_rows) != len(measurement_rows)):
        print("ERROR: len(estimate_rows) != len(measurement_rows)")

    if (len(variance_rows) != len(measurement_rows)):
        print("ERROR: len(variance_rows) != len(measurement_rows)")

    if (len(measurement_rows[0]) != jacobian.shape[0]):
        print("ERROR: len(measurement_rows[0]) != jacobian.shape[0]")

    if (len(variance_rows[0]) != jacobian.shape[0]):
        print("ERROR: len(variance_rows[0]) != jacobian.shape[0]")

    #if (len(voltageVector) != jacobian.shape[1]):
        #print(len(voltageVector))
        #print(jacobian.shape[1])
        #print("ERROR: len(voltageVector) != jacobian.shape[1]")

    numStateVariables = jacobian.shape[1]


    #i = 1 # za sada, ovdje je potrebno dobaviti i staviti nekako iGraph
    i = iGraphVector[0]
    i = tf.cast(i, tf.int32)

    measurementLabels = tf.gather(measurement_rows, i)
    estimatesWLS = tf.gather(estimate_rows, i)
    variances = tf.gather(variance_rows, i)

    measurementLabels = tf.cast(measurementLabels, tf.float64)
    estimatesWLS = tf.cast(estimatesWLS, tf.float64)
    variances = tf.cast(variances, tf.float64)

    #measurementLabels = np.array([float(x) for x in measurement_rows[i]])
    #estimatesWLS = np.array([float(x) for x in estimate_rows[i]])
    #variances = np.array([float(x) for x in variance_rows[i]])

    jacobian = tf.convert_to_tensor(jacobian)
    jacobian = tf.cast(jacobian, tf.float64)
    voltageVector = tf.cast(voltageVector, tf.float64)

    measurementFunctionsGNN = tf.matmul(jacobian, tf.transpose(voltageVector))
    
    WRSS_GNN = 0
    WRSS_GNN = tf.convert_to_tensor(WRSS_GNN)
    WRSS_GNN = tf.cast(WRSS_GNN, tf.float64)

    residuals = tf.subtract(measurementLabels, measurementFunctionsGNN)
    residualsSquares = tf.math.multiply(residuals, residuals)
    weightedResidualsSquares = tf.divide(residualsSquares, variances)

    WRSS_GNN = tf.reduce_sum(weightedResidualsSquares)

    #for iMeasurement in range(len(measurementLabels)):
        #WRSS_GNN += (measurementLabels[iMeasurement] - measurementFunctionsGNN[iMeasurement])**2 / variances[iMeasurement]

    return WRSS_GNN / numStateVariables
    #return tf.convert_to_tensor(WRSS_GNN / numStateVariables)





def evaluation_metric(y_true, y_pred):
    return WRSSLoss(y_true, y_pred)