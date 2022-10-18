Pip installing pcl for python does not work at the moment. Instead, we'll use a conda environment with python 3.6. 

1. Follow this link to install conda:
    - https://gist.github.com/kauffmanes/5e74916617f9993bc3479f401dfec7da

2. Update conda
    - Verify installation: $ conda --version
    - $ conda update conda

3. Create and manage environments
    - $ conda create --ENVIRONMENT_NAME python=3.6
    - $ conda activate ENVIRONMENT_NAME
    - To list environments:
        $ conda info --envs
    - To change back to default environment: 
        $ conda activate

4. Install other libraries
    - $ conda install numpy
    - $ conda install cython
    - Optional: 
        $ conda install matplotlib 

5. Install pcl library
    - $ sudo apt-get update
    - $ sudo apt-get install pcl-tools
    - $ sudo apt-get install libpcl-dev -y
    - $ conda config --add channels conda-forge
    - $ conda install -c sirokujira python-pcl
    - $ conda install -c jithinpr2 gtk3
    - $ conda install -y ipython
    - Go to anaconda environment lib folder:
        $ cd ~/anaconda3/envs/YOUR_ENV_NAME/lib
    - $ ln -s libboost_system.so.1.64.0 libboost_system.so.1.54.0
    - $ ln -s libboost_filesystem.so.1.64.0 libboost_filesystem.so.1.54.0
    - $ ln -s libboost_thread.so.1.64.0 libboost_thread.so.1.54.0
    - $ ln -s libboost_iostreams.so.1.64.0 libboost_iostreams.so.1.54.0

6. If any libraries are missing when running python code:
    - $ conda install LIBRARY_NAME
