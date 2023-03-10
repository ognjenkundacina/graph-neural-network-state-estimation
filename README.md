# State Estimation in Electric Power Systems Leveraging Graph Neural Networks

The goal of the state estimation (SE) algorithm is to estimate complex bus voltages as state variables based on the available set of measurements in the power system. Because phasor measurement units (PMUs) are increasingly being used in transmission power systems, there is a need for a fast SE solver that can take advantage of high sampling rates of PMUs. This research proposes training a graph neural network (GNN) to learn the estimates given the PMU voltage and current measurements as inputs, with the intent of obtaining fast and accurate predictions during the evaluation phase. GNN is trained using synthetic datasets, created using [JuliaGrid](https://github.com/mcosovic/JuliaGrid.jl) package by randomly sampling sets of measurements in the power system and labelling them with a solution obtained using a linear WLS SE with PMUs solver.

<!--More information is provided in [Robust and Fast Data-Driven Power System State Estimator Using Graph Neural Networks](https://arxiv.org/pdf/2206.02731.pdf).-->

This code can easily be adapted to the nonlinear state estimation problem considering both PMU and legacy (SCADA) measurements. For more details, please visit [Distributed Nonlinear State Estimation in Electric Power Systems using Graph Neural Networks](https://arxiv.org/pdf/2207.11465.pdf).


For the implementation of our work, we used the [IGNNITION](https://ignnition.org/) package, along with the [networkx](https://networkx.org/) package to implement the creation and augmentation logic for training and test graphs.

*"IGNNITION is the ideal framework for users with no experience in neural network programming (e.g., TensorFlow, PyTorch). With this framework, users can design and run their own Graph Neural Networks (GNN) in a matter of a few hours."*

## Installation
To install the required packages, we suggest using the popular package manager [conda](https://conda.io) package manager, in conjunction with Python 3.8. If you have not installed conda, we recommend downloading and installing 
[miniconda](https://docs.conda.io/en/latest/miniconda.html). 


After successfully installing miniconda, to open the Anaconda Prompt in Windows, you can follow these steps:

1. Press the "Windows" key on your keyboard or click on the Windows logo in the bottom left corner of your screen.
2. Type "Anaconda Prompt" in the search bar.
3. Click on the "Anaconda Prompt" app that appears in the search results.

Alternatively, you can navigate to the "Anaconda3" folder in your Windows start menu and click on "Anaconda Prompt" from there.

For Linux users, open your terminal emulator of choice (such as gnome-terminal or xterm) to access the terminal.

After opening the Anaconda Prompt (Windows), or Terminal (Linux), run the following:

```
conda create -n ignnition_environment python=3.8
conda activate ignnition_environment
```

This will create the conda environment with `ignnition_environment` name and will activate it. To install the required python libraries in the ignnition environment, please follow the steps listed below.

To download the repository from GitHub to your computer, follow these steps:

1. Click the green "Code" button located on the right side of the screen of the GitHub page of this repository.
2. Click "Download ZIP" to download the repository as a ZIP file.
3. Save the ZIP file to a directory on your computer, and extract the contents of the ZIP file to that same directory.

Once you have extracted the contents of the ZIP file, open the Anaconda Prompt on your computer and navigate to the directory where the code is located. For example, if the code is saved in a directory called "graph-neural-network-state-estimation" on your desktop, you can navigate to it by entering the following command into the Anaconda Prompt:

```
cd C:\Users\<your_username>\Desktop\graph-neural-network-state-estimation
```

After navigating to the directory, you can install the necessary packages using *PyPI* by entering the following command into the Anaconda Prompt:

```
pip install -r requirements.txt
```

This will install all the required packages specified in the "requirements.txt" file. Once the installation is complete, you can proceed to run the code by following the instructions provided in the documentation.

## Run training and inference
Various training options can be set in train_options.yaml:

- train_dataset: The path to the training dataset.

- validation_dataset: The path to the validation dataset.

- predict_dataset: The path to the testing dataset used for prediction.

- load_model_path: The path to the model weights to be loaded before training. This option can be used to resume training from a previous checkpoint.

- loss: The loss function used during training. It is set to "MeanSquaredError" by default.

- optimizer: The optimization algorithm used during training. It is set to "Adam" by default with a learning rate of 0.0004. The "clipnorm" and "clipvalue" options are used for gradient clipping.

- metrics: The evaluation metric(s) used during training. It is set to "MeanAbsoluteError" by default.

- batch_size: The batch size used during training. It is set to 32 by default.

- epochs: The maximum number of training epochs.

- epoch_size: The number of samples per epoch. If this option is not set, the entire training dataset will be used per epoch.

- shuffle_training_set: If set to "True", the training dataset will be shuffled before each epoch.

- shuffle_validation_set: If set to "True", the validation dataset will be shuffled before each epoch.

- val_samples: The number of samples used during validation. It is set to 100 by default.

- val_frequency: The frequency at which the validation is performed. It is set to 1, which means validation is performed after each epoch.

- execute_gpu: If set to "True", the code will be executed on a GPU.

- batch_norm: The type of batch normalization used during training. It is set to "mean" by default.


The training process can be started by running the following command in Anaconda Prompt:
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
Measurement data for our training, test, and validation samples are obtained using the [Measurement Generator](https://mcosovic.github.io/JuliaGrid.jl/stable/man/generator/) functionality of the [JuliaGrid](https://github.com/mcosovic/JuliaGrid.jl) package. The State Estimation problem is then solved for all of the generated samples using the [Linear State Estimation with PMUs](https://mcosovic.github.io/JuliaGrid.jl/stable/man/tbestimate/#linearpmuse), and the solutions are used for the Graph Neural Network training. The generated data (csv format) can be found in the **data_from_wls_se_solver directory**.

However, it is necessary  to structure the inputs and the outputs of every sample as graphs, using the networkx package. Run the following script to transform train, validation, and test sets from tabular to the factor graph format and to perform additional graph and feature augmentation, including binary index encoding:

```
python generate_networkx_graphs.py
```

The data, transformed into the JSON format, can be found in the **data** directory. 

As an example, we uploaded the datasets generated with measurement variances of 10<sup>-5</sup>. Due to file size limits, we could not upload the training set with 10000 samples, so the test set MSE is slightly higher than the one reported in subsection A of the paper.

## Citing
If you have found this work useful, we would appreciate citations to the [following paper](https://ieeexplore.ieee.org/document/9810559):

**<u>Plain text:</u>**

O. Kundacina, M. Cosovic and D. Vukobratovic, "State Estimation in Electric Power Systems Leveraging Graph Neural Networks," 2022 17th International Conference on Probabilistic Methods Applied to Power Systems (PMAPS), 2022, pp. 1-6, doi: 10.1109/PMAPS53380.2022.9810559.

**<u>BibTeX:</u>**
```
@INPROCEEDINGS{
    9810559,  
    author={Kundacina, Ognjen and Cosovic, Mirsad and Vukobratovic, Dejan},  
    booktitle={2022 17th International Conference on Probabilistic Methods Applied to Power Systems (PMAPS)},   
    title={State Estimation in Electric Power Systems Leveraging Graph Neural Networks},   
    year={2022},  
    volume={},  
    number={},  
    pages={1-6},  
    doi={10.1109/PMAPS53380.2022.9810559}
}

```


... or to our nonlinear state estimation paper:


**<u>Plain text:</u>**

O. Kundacina, M. Cosovic, D. Miskovic and D. Vukobratovic, "Distributed Nonlinear State Estimation in Electric Power Systems using Graph Neural Networks," 2022 IEEE International Conference on Smart Grid Communications (SmartGridComm), 2022, pp. 1-6.

**<u>BibTeX:</u>**
```
@INPROCEEDINGS{
    9810559,  
    author={Kundacina, Ognjen and Cosovic, Mirsad and Miskovic, Dragisa and Vukobratovic, Dejan},  
    booktitle={2022 IEEE International Conference on Smart Grid Communications (SmartGridComm)},   
    title={Distributed Nonlinear State Estimation in Electric Power Systems using Graph Neural Networks},   
    year={2022},  
    volume={},  
    number={},  
    pages={1-6}
}

```
