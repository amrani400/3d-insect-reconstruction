<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    

</head>
<body>
    <header>
        <h1>Dynamic Adaptive Sampling for Accurate Image-based 3D Insect Reconstruction Using Neural Implicit Surfaces (accepted for DICTA 2025)
</h1>
    </header>
    <nav>
        <ul>
            <li><a href="#overview">Overview</a></li>
            <li><a href="#installation">Installation</a></li>
            <li><a href="#usage">Usage</a></li>
            <li><a href="#results">Results</a></li>
        </ul>
    </nav>
    <section id="overview">
        <h2>Overview</h2>
        <p>3D Insect Reconstruction is a framework for generating high-fidelity 3D models of insects from multi-view images. Built on neural implicit surfaces, it uses dynamic adaptive sampling to capture intricate details like translucent wings and delicate antennae. Key features include:</p>
        <ul>
            <li>Thin structure scores to identify critical regions.</li>
            <li>Hybrid ray sampling for efficient coverage.</li>
            <li>Score-weighted color loss for detail preservation.</li>
            <li>Adaptive refinement for enhanced accuracy.</li>
        </ul>
        <p>This repository provides scripts and configurations to preprocess data, train models, and generate 3D reconstructions, optimized for datasets of insects like moths, true bugs, and butterflies.</p>
    </section>
    <section id="installation">
        <h2>Installation</h2>
        <p>The project uses a Conda environment with Python 3.8 or later, mirroring the setup of <a href="https://github.com/Totoro97/NeuS">NeuS</a>. Follow these steps to set up the environment.</p>
        <h3>Prerequisites</h3>
        <ul>
            <li>Conda (Miniconda or Anaconda)</li>
            <li>NVIDIA GPU with CUDA support</li>
            <li>Git</li>
        </ul>
        <h3>Steps</h3>
        <ol>
            <li><strong>Clone the Repository</strong>:<br>
                <code>git clone https://github.com/a93088428/3d-insect-reconstruction.git</code><br>
                <code>cd 3d-insect-reconstruction</code>
            </li>
            <li><strong>Create a Conda Environment</strong>:<br>
                <code>conda create -n insect_recon python=3.8</code><br>
                <code>conda activate insect_recon</code>
            </li>
            <li><strong>Install PyTorch</strong>:<br>
                Install PyTorch with CUDA support (e.g., CUDA 11.8):<br>
                <code>pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118</code><br>
                Check <a href="https://pytorch.org/get-started/locally/">PyTorch's website</a> for your system’s version.
            </li>
            <li><strong>Install PyTorch3D</strong>:<br>
                <code>pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"</code><br>
                See <a href="https://github.com/facebookresearch/pytorch3d#installation">PyTorch3D installation guide</a> for pre-built wheels.
            </li>
            <li><strong>Install Additional Dependencies</strong>:<br>
                <code>pip install numpy opencv-python pillow scipy tqdm scikit-image</code>
            </li>
            <li><strong>Verify Installation</strong>:<br>
                <code>python -c "import torch; import pytorch3d; print('Setup complete!')"</code>
            </li>
        </ol>
    </section>
    <section id="usage">
        <h2>Usage</h2>
        <p>This repository includes scripts to preprocess data, train models, and generate 3D reconstructions.</p>
        <h3>Directory Structure</h3>
        <ul>
            <li><code>confs/</code>: Configuration files for experiments.</li>
            <li><code>models/</code>: Pre-trained models or checkpoints.</li>
            <li><code>DNASC.py</code>: Core script for dynamic adaptive sampling and reconstruction.</li>
            <li><code>exp_runner.py</code>: Script to run experiments.</li>
            <li><code>all_meshes.png</code>: Example reconstructed meshes.</li>
            <li><code>score_map.png</code>: Visualization of score map computation.</li>
        </ul>
        <h3>Running an Experiment</h3>
        <ol>
            <li><strong>Prepare Data</strong>:<br>
                Place multi-view insect images in a dataset folder (e.g., COLMAP or DTU format). Update the configuration in <code>confs/</code>.
            </li>
            <li><strong>Run Training</strong>:<br>
                <code>python exp_runner.py --mode train --conf ./confs/your_config.conf</code>
            </li>
            <li><strong>Generate Reconstruction</strong>:<br>
                <code>python DNASC.py --mode reconstruct --conf ./confs/your_config.conf</code>
            </li>
            <li><strong>Visualize Results</strong>:<br>
                Check output meshes in the experiment directory (specified in the config).
            </li>
        </ol>
        <h3>Example</h3>
        <p>Train and reconstruct with a sample configuration:</p>
        <code>python exp_runner.py --mode train --conf ./confs/moth.conf</code><br>
        <code>python DNASC.py --mode reconstruct --conf ./confs/moth.conf</code>
    </section>
    <section id="results">
        <h2>Results</h2>
        <p>The framework produces high-fidelity 3D insect models, capturing fine details like translucent wings and antennae. Example outputs:</p>
        <figure>
            <img src="score_map.png" alt="Score map computation process">
            <figcaption>Score map computation process</figcaption>
        </figure>
        <figure>
            <img src="all_meshes.png" alt="Reconstructed insect meshes">
            <figcaption>Reconstructed insect meshes</figcaption>
        </figure>
    </section>
    <footer>
        <p>Licensed under the MIT License. See the <a href="LICENSE">LICENSE</a> file for details. Built upon the <a href="https://github.com/Totoro97/NeuS">NeuS</a> framework.</p>
    </footer>
</body>
</html>
