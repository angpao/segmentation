brew install pyenv
pyenv exec python -V
pyenv exec python -m venv .venv
source .venv/bin/activate
Create requirements.txt
pip install -r requirements.txt

Install Library
install conda

create conda environment conda create -n learn python=3.9
activate the conda environment conda activate learn
install pytorch (https://pytorch.org/get-started/locally/)

without gpu:

conda install pytorch torchvision torchaudio cpuonly -c pytorch
or with gpu:

conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
install pandas

conda install pandas
install tensorboard

pip install tensorboard
(segmentation) install albumentation

conda install -c conda-forge imgaug
conda install -c conda-forge albumentations
(seq2seq) install spacy

pip install -U spacy
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
(seq2seq) install torchtext and torch-data

conda install -c pytorch torchtext torchdata
Download Dataset
https://drive.google.com/drive/folders/1feL5X6epYQiGaT-dHpgV7yWU7_ZthVVI?usp=sharing

1.Get dataset loader
1.1 define preprocessing transform steps
1.2 create dataset object (define where to load data)
1.3 create dataloader object (define batchsize and how to load data)

2.Define model components
2.1 network
2.2 loss function
2.3 optimizer

3.Define logger object

4.Training loop
4.0 set device
4.1 make prediction
4.2 compute loss
4.3 compute gradients
4.4 adjust the weights
4.5 collect result into the logger

#git
git remote add origin git@github.com:angpao/classifier.git
git push -u origin main
