3D Insect Reconstruction
Overview
3D Insect Reconstruction is a Python-based framework for reconstructing high-fidelity 3D models of insects from multi-view images. This project leverages neural implicit surfaces with dynamic adaptive sampling to capture intricate details such as translucent wings and delicate antennae. Key features include:

Thin structure scores to identify critical regions.
Hybrid ray sampling for efficient coverage.
Score-weighted color loss for detail preservation.
Adaptive refinement for enhanced reconstruction accuracy.

This repository provides the tools to preprocess data, train models, and generate 3D reconstructions, achieving superior performance on custom datasets of insects like moths, true bugs, and butterflies.
Installation
The project requires a Conda environment with Python 3.8 or later. The dependencies are aligned with those used in NeuS. Follow the steps below to set up the environment.
Prerequisites

Conda (recommended: Miniconda or Anaconda)
NVIDIA GPU with CUDA support (for PyTorch and PyTorch3D)
Git

Steps

Clone the Repository:
git clone https://github.com/a93088428/3d-insect-reconstruction.git
cd 3d-insect-reconstruction


Create a Conda Environment:
conda create -n insect_recon python=3.8
conda activate insect_recon


Install PyTorch:Install PyTorch with CUDA support. For example, for CUDA 11.8:
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

Check the PyTorch website for the appropriate version for your system.

Install PyTorch3D:PyTorch3D is required for 3D operations. Install it using:
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

Alternatively, use a pre-built wheel if available for your CUDA version (see PyTorch3D installation guide).

Install Additional Dependencies:Install the remaining dependencies:
pip install numpy opencv-python pillow scipy tqdm scikit-image


Verify Installation:Ensure all dependencies are correctly installed by running:
python -c "import torch; import pytorch3d; print('Setup complete!')"



Usage
The repository includes scripts and configurations to preprocess data, train the model, and generate 3D reconstructions.
Directory Structure

confs/: Configuration files for different experiments.
models/: Pre-trained models or model checkpoints.
DNASC.py: Core script for dynamic adaptive sampling and reconstruction.
exp_runner.py: Script to run experiments with specified configurations.
all_meshes.png: Example output showing reconstructed meshes.
score_map.png: Visualization of the score map computation process.

Running an Experiment

Prepare Data:

Place your multi-view insect images in a dataset folder (format compatible with NeuS, e.g., COLMAP or DTU format).
Update the configuration file in confs/ with the dataset path and parameters.


Run Training:Use exp_runner.py to train the model:
python exp_runner.py --mode train --conf ./confs/your_config.conf

Replace your_config.conf with the appropriate configuration file.

Generate Reconstruction:After training, generate the 3D mesh:
python DNASC.py --mode reconstruct --conf ./confs/your_config.conf


Visualize Results:Check the output meshes in the experiment directory (specified in the config file). Example results are shown in all_meshes.png.


Example
To train and reconstruct using a sample configuration:
python exp_runner.py --mode train --conf ./confs/moth.conf
python DNASC.py --mode reconstruct --conf ./confs/moth.conf

Results
The framework achieves high-fidelity 3D reconstructions of insects, capturing fine details like translucent wings and delicate antennae. Below are example outputs:

Score Map: Visualizes the computation of thin structure scores for adaptive sampling (see score_map.png).
Reconstructed Meshes: Comparison of reconstructed insect meshes against baselines (see all_meshes.png).

Contributing
Contributions are welcome! Please open an issue or submit a pull request for bug fixes, feature additions, or improvements.
License
This project is licensed under the MIT License. See the LICENSE file for details.
Acknowledgments
This project builds upon the NeuS framework. We thank the authors for their foundational work in neural surface reconstruction.
