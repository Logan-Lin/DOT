# DOT: Diffusion-based Origin-destination Travel Time Estimation

Implementation code for the model **Diffusion-based Origin-destination Travel Time Estimation (DOT)** proposed in the SIGMOD 2024 paper *Origin-Destination Travel Time Oracle for Map-based Services*.

## Model overview

![overall-framework](./assets/overall-framework.jpg)

DOT follows a two-stage framework. The first stage is the Pixelated Trajectory (PiT) inference stage, and the second stage is the PiT travel time estimation stage.

In the PiT inference stage, DOT infers the PiT corresponds to a given ODT-Input. The ODT-Input is considered the conditional information incorporated into a conditioned PiT denoiser. The denoiser samples from standard Gaussian noise at the beginning. Then, it produces the inferred PiT conditioned on the ODT-Input through a multi-step conditioned denoising diffusion process.

In the PiT travel time estimation stage, DOT estimates the travel time based on the inferred PiT. The PiT is first flattened and mapped into a feature sequence to capture the global spatial-temporal correlation better. A Masked Vision Transformer (MViT) is used to estimate the travel time. 

## Code Walkthrough

### Runtime Environment

Assume `python3` is already installed on your computer, and you have navigated to the root directory of the project. To create a virtual environment, execute the following commands:

```bash
python3 -m venv ./venv
source ./venv/bin/activate
```

This will set up a Python virtual environment in the `./venv` directory. You can specify a different directory if needed. To exit the environment, use the `deactivate` command. Alternatively, other solutions like Anaconda, Docker, or Nix can be used.

Next, install the required dependencies for this project by running:

```bash
pip3 install -r requirements.txt
```

### Preparing the Dataset

It is recommended to prepare the dataset before starting the main process. Some metadata can be cached in files to avoid the need for preprocessing each time the main process runs.

Sample trajectory datasets are provided in the `sample` directory. These datasets follow the same format as the full dataset but are smaller in size, making them ideal for local debugging.

For instance, to pre-process the `chengdu` dataset into PiT format using a $20 \times 20$ grid, run the following command:

```bash
python3 dataset.py --trajpath ./sample/chengdu.h5 --metadir ./meta/chengdu -s 20 -t image
```

This will:

- Load the raw trajectory data from `./sample/chengdu.h5`
- Pre-process the trajectory data using a $20 \times 20$ grid
- Calculate OD pairs, ETA ground truth, and PiT for each trajectory
- Store the trajectory pairs for training, validation, and testing as a NumPy binary file `S20_FFalse_image.npz` in the `./meta/chengdu` directory.

Other types of metadata (e.g., flattened PiT, trajectory data) can also be cached by adjusting the parameters:

```bash
# For Trajectory Data
python3 dataset.py --trajpath ./sample/chengdu.h5 --metadir ./meta/chengdu -s 20 -t traj

# For Flattened PiT
python3 dataset.py --trajpath ./sample/chengdu.h5 --metadir ./meta/chengdu -s 20 -t image --flat
```

### Training and Evaluation Pipeline

The `main.py` script handles the training and testing processes of DOT. Parameters are passed as command-line arguments. For example, to train the model on the `chengdu` dataset with $20 \times 20$ grids for 50 epochs, run the following command:

```bash
python3 main.py --trajpath ./sample/chengdu.h5 --metadir ./meta/chengdu -s 20 --traindiff -e 50 --device cuda:0
```

This will:

- Load the cached metadata prepared earlier
- Initialize the DOT model, including the diffusion processes and the Unet-based denoiser
- Train the denoiser by supervising the reverse diffusion process
- Visualize generated and ground truth PiTs on the validation set
- Train the ETA model based on the PiTs

Any hyperparameters not explicitly provided will use their default values.

## Paper Information

Reference:

> Yan Lin, Huaiyu Wan, Jilin Hu, Shengnan Guo, Bin Yang, Youfang Lin, Christian S. Jensen. Origin-Destination Travel Time Oracle for Map-based Services. Proceedings of the ACM on Management of Data 1.3 (2023): 1-27.

Paper: https://dl.acm.org/doi/10.1145/3617337

Pre-print: https://arxiv.org/abs/2307.03048

If you have any further questions, feel free to contact me directly. My contact information is available on my homepage: https://www.yanlincs.com/