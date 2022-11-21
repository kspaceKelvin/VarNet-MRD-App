#  Variational Network (VarNet) Image Reconstruction
This code implements a variational network for image reconstruction as described by Sriram A *et al* in [End-to-End Variational Networks for Accelerated MRI Reconstruction](https://arxiv.org/abs/2004.06688).

## Input Data
This app takes as undersampled raw k-space data as input and outputs VarNet reconstructed images.  Auto-calibration signal (ACS) data from a fully-sampled central k-space with integrated reference lines is required.  Multi-slice data are supported.

## Supported Configurations
The ``varnet`` config is used for this workflow.  The acceleration rate and size of the ACS region can be retrospectively downsampled by appending ``_Rx`` or ``_ACSy`` where ``x`` is the acceleration rate and ``y`` is the number of ACS lines.  For example, a config of ``varnet_R8_ACS24`` indicates retrospective downsampling to rate 8 acceleration with 24 ACS lines.

## Running the app
This MRD app can be downloaded from Docker Hub at https://hub.docker.com/r/kspacekelvin/varnet-mrd-app.  In a command prompt on a system with [Docker](https://www.docker.com/) installed, download the Docker image:
```
docker pull kspacekelvin/varnet-mrd-app
```

Start the Docker image and share port 9002:
```
docker run --rm -p 9002:9002 kspacekelvin/varnet-mrd-app
```

In another window, use an MRD client such as the one provided from the [python-ismrmrd-server](https://github.com/kspaceKelvin/python-ismrmrd-server#11-reconstruct-a-phantom-raw-data-set-using-the-mrd-clientserver-pair):

Run the client and send the data to the server.  For downsampling to R=8 and 24 ACS lines (similar to the fastMRI dataset):
```
python3 client.py -o raw.mrd -c varnet_R8_ACS24 img.mrd
```

The output file (e.g. img.mrd) contains reconstructed images from all slices in a single group.
