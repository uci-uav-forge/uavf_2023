FROM ros:noetic

WORKDIR /root/catkin_ws/src/uavf_2023
COPY . /root/catkin_ws/src/uavf_2023

RUN apt-get update

# opencv dependencies (https://stackoverflow.com/questions/55313610/importerror-libgl-so-1-cannot-open-shared-object-file-no-such-file-or-directo)
RUN apt-get install -y ffmpeg libsm6 libxext6

RUN apt-get install -y python3-pip && \
    apt-get install -y ros-noetic-mavros ros-noetic-mavros-extras

# utils (not strictly necessary)
RUN apt-get install -y tmux vim

# comment this out of you have a GPU
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu

RUN ./install_geographiclib_datasets.sh

RUN pip3 install -r requirements.txt
