# State Estimation in Electric Power Systems Leveraging Graph Neural Networks

The goal of the state estimation (SE) algorithm is to estimate complex bus voltages as state variables based on the available set of measurements in the power system. Because phasor measurement units (PMUs) are increasingly being used in transmission power systems, there is a need for a fast SE solver that can take advantage of high sampling rates of PMUs. This research proposes training a graph neural network (GNN) to learn the estimates given the PMU voltage and current measurements as inputs, with the intent of obtaining fast and accurate predictions during the evaluation phase. GNN is trained using synthetic datasets, created using [JuliaGrid](https://github.com/mcosovic/JuliaGrid.jl) package by randomly sampling sets of measurements in the power system and labelling them with a solution obtained using a linear WLS SE with PMUs solver.

More information is provided in [Robust and Fast Data-Driven Power System State Estimator Using Graph Neural Networks](https://arxiv.org/pdf/2206.02731.pdf).

For the implementation of our work, we used the [IGNNITION](https://ignnition.org/) package, along with the [networkx](https://networkx.org/) package to implement the creation and augmentation logic for training and test graphs.

*"IGNNITION is the ideal framework for users with no experience in neural network programming (e.g., TensorFlow, PyTorch). With this framework, users can design and run their own Graph Neural Networks (GNN) in a matter of a few hours."*

## Installation
To install the necessary packages, we recommend using the [conda](https://conda.io) package manager, along with Python 3.8. First install 
[miniconda](https://docs.conda.io/en/latest/miniconda.html). Then open the Anaconda Prompt (Windows), or Terminal (Linux) and then run the following:

```
conda create -n ignnition_environment python=3.8
conda activate ignnition_environment
```

This will create the conda environment with `ignnition_environment` name and will activate it. Navigate to the directory containing the code downloaded from this repository, and install the necessary packages with the following command using *PyPI*:

```
pip install -r requirements.txt
```

## Run training and inference
Various training options can be set in train_options.yaml. The training process can be started by running:
```
python train.py
```
Running the script generates the **CheckPoint** directory in which trained model parameters are saved after each epoch. Also, the training and validation set loss is logged in the Anaconda Prompt/Terminal after each epoch. After the training process is finished, it is recommended to select the model from the epoch which resulted in the lowest validation set loss. To run the inference based on the model from, for example, the 15th epoch, the following line needs to be added to the train_options.yaml:
```
load_model_path: ./CheckPoint/experiment_2022_05_31_21_38_17/ckpt/weights.15-0.000
```
Finally, the inference on the whole test set, along with plotting the more detailed results for one test sample, can be run using:
```
python predict.py
```

## Dataset generation
Measurement data for our training, test, and validation samples are obtained using the [Measurement Generator](https://mcosovic.github.io/JuliaGrid.jl/stable/man/generator/) functionality of the [JuliaGrid](https://github.com/mcosovic/JuliaGrid.jl) package. The State Estimation problem is then solved for all of the generated samples using the [Linear State Estimation with PMUs](https://mcosovic.github.io/JuliaGrid.jl/stable/man/tbestimate/#linearpmuse), and the solutions are used for the Graph Neural Network training. The generated data can be found in the **data_from_wls_se_solver directory**.

However, it is necessary  to structure the inputs and the outputs of every sample as graphs, using the networkx package. The transformed data can be found in the **data** directory. 

**We cannot share the code that generates the factor graph and the augmented factor graph yet because that part of the algorithm is in the process of being patented.**

As an example, we uploaded the datasets generated with measurement variances of 10<sup>-5</sup>. Due to file size limits, we could not upload the training set with 10000 samples, so the test set MSE is slightly higher than the one reported in subsection A of the paper.

## Citing
If you have found this work useful, we would appreciate citations to the following paper:

**<u>Plain text:</u>**

O. Kundacina, M. Cosovic, and D. Vukobratovic, “State estimation in electric power systems leveraging graph neural networks,” in International Conference on Probabilistic Methods Applied to Power Systems (PMAPS), 2022.

**<u>BibTeX:</u>**
```
 @INPROCEEDINGS{
  kundacina2022state,  
  author={Kundacina, Ognjen and Cosovic, Mirsad and Vukobratovic, Dejan},  
  booktitle={International Conference on Probabilistic Methods Applied to Power Systems (PMAPS)},   
  title={State Estimation in Electric Power Systems Leveraging Graph Neural Networks},   
  year={2022}
 }


```
