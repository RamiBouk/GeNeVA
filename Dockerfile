FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Install necessary system packages
RUN apt-get update 
RUN apt-get install -y wget gnupg2 git unzip 
RUN apt-get install -y rar 
RUN rm -rf /var/lib/apt/lists/*


# Add NVIDIA CUDA repository key
RUN wget -O /tmp/NVIDIA-KEY.pub https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub && \
    apt-key add /tmp/NVIDIA-KEY.pub && \
    rm -rf /tmp/NVIDIA-KEY.pub

# Install Miniconda
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh --no-check-certificate
RUN bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda 
RUN rm Miniconda3-latest-Linux-x86_64.sh

# Add Miniconda to PATH
ENV PATH=/opt/conda/bin:$PATH

# Clone the repository
RUN git clone https://github.com/Maluuba/GeNeVA_datasets.git /opt/GeNeVA_datasets

# Change directory to the repository
WORKDIR /opt/GeNeVA_datasets


# Create a conda environment for this repository
RUN conda env create -f environment.yml

# Activate the environment
RUN echo "source activate geneva" > ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]
RUN source activate geneva

# Modify the download.sh script
RUN sed -i 's#https://vision.ece.vt.edu/clipart/dataset/AbstractScenes_v1.1.zip#https://download.microsoft.com/download/4/5/D/45D1BBFC-7944-4AC5-AED2-1A35D85662D7/AbstractScenes_v1.1.zip#' scripts/download_data.sh

# Download external data files
RUN ./scripts/download_data.sh

# Download GeNeVA data files to the repository
RUN wget  https://download.microsoft.com/download/6/D/0/6D074C80-78C0-4976-AB33-880BFA571F3C/GeNeVA-v1.zip
RUN unzip GeNeVA-v1.zip 
RUN rar x GeNeVA-v1/data.rar
RUN rar x GeNeVA-v1/CoDraw_images.rar raw-data/CoDraw
RUN rar x GeNeVA-v1/i-CLEVR.rar raw-data

RUN sed -i "36s/with open(original_glove, 'r')/with open(original_glove, 'r', encoding='utf-8')/" scripts/joint_codraw_iclevr/generate_glove_file.py

RUN python scripts/joint_codraw_iclevr/generate_glove_file.py

RUN python -c "import nltk; nltk.download('punkt')"

RUN python scripts/codraw_dataset_generation/codraw_add_data_to_raw.py
RUN python scripts/codraw_dataset_generation/codraw_raw_to_hdf5.py  
RUN python scripts/codraw_dataset_generation/codraw_object_detection.py 


RUN python scripts/iclevr_dataset_generation/iclevr_add_data_to_raw.py 
RUN python scripts/iclevr_dataset_generation/iclevr_raw_to_hdf5.py      
RUN python scripts/iclevr_dataset_generation/iclevr_object_detection.py  
    # Clone the GeNeVA repository

RUN git clone https://github.com/Maluuba/GeNeVA.git /opt/GeNeVA

# Change directory to the repository
WORKDIR /opt/GeNeVA

# Add conda-forge channel
RUN conda config --add channels defaults && \
    conda config --add channels conda-forge && \
    conda config --add channels pytorch && \
    conda config --add channels pytorch-lts && \
    conda config --add channels soumith && \
    conda config --add channels engility && \
    conda config --add channels kitware-danesfield-pt && \
    conda config --add channels dhirschfeld && \
    conda config --add channels gaiar

RUN sed -i 's#geneva#geneva2#' environment.yml 

# Create a conda environment for this repository
RUN conda env create -f environment.yml

# Activate the environment
RUN echo "source activate geneva2" > ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

# Run setup
RUN python setup.py install

# Train the object detector and localizer for i-CLEVR
#RUN python scripts/traiJ_object_detector_localizer.py --num-classes=24 --train-hdf5=../GeNeVA_datase:qts/data/iCLEVR/clevr_obj_train.h5 --valid-hdf5=../GeNeVA_datasets/data/iCLEVR/clevr_obj_val.h5 --cuda-enabled
#RUN visdom & 
#RUN sleep 20 

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Train the object detector and localizer for CoDraw
#RUN cp scripts/train_object_detector_localizer.py test_run.py
ENV PYTHONPATH="/opt/GeneVA/geneva:${PYTHONPATH}"

EXPOSE 8097
# to ssh into the docker
EXPOSE 22

RUN apt update && apt install -y vim
RUN echo "python ./scripts/train_object_detector_localizer.py --num-classes=58 --train-hdf5=../GeNeVA_datasets/data/CoDraw/codraw_obj_train.h5 --valid-hdf5=../GeNeVA_datasets/data/CoDraw/codraw_obj_val.h5 --cuda-enabled" >> run.sh

RUN echo 'pip install torch==1.8.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html'\
    >> install_correct_pytorch.sh
# Install SSH server
RUN apt-get update && \
    apt-get install -y openssh-server && \
    mkdir /var/run/sshd

# Set root password
RUN echo 'root:password' | chpasswd

# Allow SSH root login
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# Allow SSH access from any IP address
RUN sed -i 's/#ListenAddress 0.0.0.0/ListenAddress 0.0.0.0/' /etc/ssh/sshd_config

# Expose SSH port
EXPOSE 22

# Start SSH server
CMD ["/usr/sbin/sshd", "-D"]


