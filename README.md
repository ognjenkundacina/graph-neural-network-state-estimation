# State Estimation in Electric Power Systems Leveraging Graph Neural Networks

The goal of the state estimation (SE) algorithm is to estimate complex bus voltages as state variables based on the available set of measurements in the power system. Because phasor measurement units (PMUs) are increasingly being used in transmission power systems, there is a need for a fast SE solver that can take advantage of high sampling rates of PMUs. This research proposes training a graph neural network (GNN) to learn the estimates given the PMU voltage and current measurements as inputs, with the intent of obtaining fast and accurate predictions during the evaluation phase. GNN is trained using synthetic datasets, created using [JuliaGrid](https://github.com/mcosovic/JuliaGrid.jl) package by randomly sampling sets of measurements in the power system and labelling them with a solution obtained using a linear WLS SE with PMUs solver.

More information is provided in [State Estimation in Electric Power Systems Leveraging Graph Neural Networks](https://arxiv.org/abs/2201.04056).

For the implementation of our work, we used the [IGNNITION](https://ignnition.org/) package, along with the [networkx](https://networkx.org/) package to incororate the creation and augmentation logic for training and test graphs.

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

## Citing
If you have found this work useful, we would appreciate citations to the following paper:

**<u>Plain text:</u>**

O. Kundacina, M. Cosovic, and D. Vukobratovic, “State estimation in electric power systems leveraging graph neural networks,” arXiv preprint arXiv:2201.04056, 2022.

**<u>BibTeX:</u>**
```
@misc{kundacina2022state,
  title={State Estimation in Electric Power Systems Leveraging Graph Neural Networks},
  author={Kundacina, Ognjen and Cosovic, Mirsad and Vukobratovic, Dejan},
  journal={arXiv preprint arXiv:2201.04056},
  year={2022}
}

```
