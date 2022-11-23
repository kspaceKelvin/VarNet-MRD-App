FROM kspacekelvin/fire-python

# CPU-only PyTorch
RUN pip3 install --no-cache-dir torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
RUN apt-get update && apt-get install --no-install-recommends -y git curl

# Build from latest fastMRI instead of using the pip package as the current code is not compatible
RUN cd /opt/code && \
    git clone https://github.com/facebookresearch/fastMRI.git && \
    cd fastMRI && \
    pip3 install --no-cache-dir .

# Download this repository and merge with python-ismrmrd-server
RUN cd /opt/code && \
    git clone https://github.com/kspaceKelvin/VarNet-MRD-App.git && \
    cd VarNet-MRD-App && \
    cp -f *.py ../python-ismrmrd-server

# Pre-trained fastMRI VarNet models: https://github.com/facebookresearch/fastMRI
RUN mkdir -p /opt/code/python-ismrmrd-server/models

# Copy from local cache if exists
COPY *models/varnet_brain_leaderboard_state_dict.pt /opt/code/python-ismrmrd-server/models
COPY *models/varnet_knee_leaderboard_state_dict.pt  /opt/code/python-ismrmrd-server/models

# Download if not present
RUN ([ ! -f "/opt/code/python-ismrmrd-server/models/varnet_brain_leaderboard_state_dict.pt" ] && \
    curl -C - "https://dl.fbaipublicfiles.com/fastMRI/trained_models/varnet/brain_leaderboard_state_dict.pt" --output /opt/code/python-ismrmrd-server/models/varnet_brain_leaderboard_state_dict.pt) || true

RUN ([ ! -f "/opt/code/python-ismrmrd-server/models/varnet_knee_leaderboard_state_dict.pt" ] && \
    curl -C - "https://dl.fbaipublicfiles.com/fastMRI/trained_models/varnet/knee_leaderboard_state_dict.pt" --output /opt/code/python-ismrmrd-server/models/varnet_knee_leaderboard_state_dict.pt) || true

# Cleanup files not required after installation
RUN  apt-get remove git curl -y \
     && apt-get clean \
     && rm -rf /var/lib/apt/lists/* \
     && rm -rf /root/.cache/pip

# Copy test data
RUN mkdir -p /test_data
COPY *test_data/t1_tse_R6.mrd /test_data

# Set the starting directory so that code can use relative paths
WORKDIR /opt/code/python-ismrmrd-server