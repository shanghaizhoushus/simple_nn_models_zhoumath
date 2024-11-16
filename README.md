
# Simple Neural Network Models

## Overview
This project contains implementations of simple neural network models, including training scripts for PyTorch and Lightning frameworks. The goal is to provide a modular, easy-to-understand codebase for experimenting with and understanding the basics of neural network training. This repository is structured to support different frameworks, allowing users to compare their performance and flexibility.

## Project Structure
```
simple_nn_models_zhoumath/
├── .gitignore
├── LICENSE
├── requirements.txt
├── examples/
│   ├── cal_ranking_by_freq.py
│   ├── simple_nn_trainer_lightningai_examples/
│   │   └── simple_nn_trainer_lightningai_example_script.py
│   ├── simple_nn_trainer_lightning_examples/
│   │   └── simple_nn_trainer_lightning_example_script.py
│   └── simple_nn_trainer_pytorch_examples/
│       └── simple_nn_trainer_pytorch_example_script.py
├── scripts/
│   ├── simple_nn_samples.py
│   ├── simple_nn_trainer_lightning.py
│   ├── simple_nn_trainer_lightningai.py
│   └── simple_nn_trainer_pytorch.py
```

- **`.gitignore`**: Lists files and directories to be ignored by version control.
- **`LICENSE`**: Contains the licensing terms for the project.
- **`requirements.txt`**: Lists the dependencies required for the project.
- **`examples/`**: Contains example scripts demonstrating the usage of the training modules with different frameworks.
- **`scripts/`**: Contains the core scripts that define simple neural network architectures, trainers, and training processes.

## Installation
To get started, you need to install the required dependencies. Run the following command to install all required packages:

```sh
pip install -r requirements.txt
```

Note: Make sure you have Python 3.7 or higher installed.

## Usage
This repository contains different neural network models and training scripts. You can start training a simple neural network using the provided training scripts.

### Example: Training with PyTorch
You can use the PyTorch-based trainer to train a simple neural network. Navigate to the `scripts/` directory and run the following command:

```sh
python simple_nn_trainer_pytorch.py
```

This script includes all the necessary steps to train, validate, and test a neural network model.

### Example: Training with PyTorch Lightning
To train a model using PyTorch Lightning, use the `simple_nn_trainer_lightning.py` script:

```sh
python scripts/simple_nn_trainer_lightning.py
```

This script uses PyTorch Lightning to simplify the training process and manage the training loop efficiently.

## Modules Overview
- **SimpleLR**: Implements a simple learning rate scheduler.
- **SimpleNN**: A basic feedforward neural network.
- **GeneralNN**: A more flexible neural network model that can be customized with different architectures.
- **SimpleFTTransformer**: Implements a simplified Transformer-based architecture.
- **TransformerLayer**: Represents a single layer of a Transformer network.

## Example Scripts
The `examples/` directory contains scripts to demonstrate how to use the models and training modules:
- **`cal_ranking_by_freq.py`**: Calculates rankings based on frequency for a given dataset.
- **Framework-Specific Examples**: Each folder (e.g., `simple_nn_trainer_lightningai_examples`) contains an example script showing how to use the respective framework for training.

## Dependencies
Make sure the following dependencies are installed, which are listed in `requirements.txt`:
- `torch`
- `pytorch-lightning`
- `pandas`
- `matplotlib`

To add these missing dependencies, update the `requirements.txt` accordingly.

## Contributing
Contributions are welcome! If you encounter any issues or have suggestions, feel free to open an issue or submit a pull request.

## License
This project is licensed under the terms specified in the `LICENSE` file.

## Future Improvements
- Add unit tests for each module to improve code reliability.
- Enhance documentation and add more descriptive comments throughout the code.
- Improve the example scripts to include complete training, validation, and testing workflows.

## Contact
If you have questions or need further clarification, please feel free to contact the maintainers via the repository issues page.
