# Robust-GAN
This repository provides an implementation of STV-GAN.

## Environment
The following dependencies are needed:
- Python 3.7
- PyTorch 1.12.1

# Files
This repository includes the following files,
- `demo.ipynb`: A Jupyter notebook demo of the code. It includes two core examples: the estimation of the mean vector and the scatter matrix.
- `gan.py`: This script includes a class for performing robust estimation.
- `gan_torch/mu.py`: This script provides a stable implementation for performing robust mean vector estimation.

# Usage
1. Clone this repository to your local machine:

    ```bash
    git clone https://github.com/monarsan/robust-GAN
    cd robust-GAN
    ```

2. Create a new Conda environment from the `environment.yaml` file:

    ```bash
    conda env create -f environment.yaml
    ```

   This command creates a new Conda environment, which is named robust-gan.

3. Activate the new Conda environment:

    ```bash
    conda activate robust-gan
    ```

4. You can now start running demo.ipynb.