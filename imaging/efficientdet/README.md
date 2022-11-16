Commands to run before (supplied by https://colab.research.google.com/drive/1I-g80m8xv3EP3J9VeL8qG-gHX2x7t1ma?usp=sharing#scrollTo=c4MfM8POBQtw):
SETTING UP EFFICIENTDET:
git clone --depth 1 https://github.com/google/automl
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
(in automl/efficientdet):
    pip install -r requirements.txt
    BEFORE RUNNING THE NEXT COMMAND
    select a backbone name of the form 'efficientnet-bx' where x is 0-6 inclusive
    wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/ckptsaug/{backbone_name}.tar.gz
DOWNLOADING DATA:
curl -o comp20k.zip -L 'https://www.dropbox.com/s/fpvoj2mblnr3567/comp20k.zip?dl=0' > comp20k.zip; unzip comp20k.zip; rm comp20k.zip
SETTING UP TRAIN:
copy the config text from the notebook into automl/efficientdet/configs/hparams.yaml
pip install tensorflow_addons
add this shit to your bashrc if tensorflow isn't using GPU:
> aside: "add to your bashrc" means the command will be automatically executed every time you open a terminal with bash as your shell. Usually a bashrc is a file with the path ~/.bashrc. You could also just run the command before running the script
`export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/`
Run this command from automl/efficientdet to start the train:
```
python -m tf2.train --mode=traineval \
    --train_file_pattern=./comp20k/train/20k.tfrecord \
    --val_file_pattern=./comp20k/valid/20k.tfrecord \
    --model_name=efficientdet-d0 \
    --model_dir=./shapev5 \
    --batch_size=25 \
    --use_xla=False \
    --eval_samples=4000 \
    --num_examples_per_epoch=160000  --num_epochs=300 \
    --hparams=./configs/hparams.yaml
    #--pretrained_ckpt=put the checkpoint path here. The checkpoint is the folder containing the various checkpoint files. \
```
