git clone https://github.com/gabrielmittag/NISQA.git
wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.2-Linux-x86_64.sh
chmod +x Miniconda3-py37_4.8.2-Linux-x86_64.sh
bash ./Miniconda3-py37_4.8.2-Linux-x86_64.sh -b -f -p /usr/local
conda env create -f NISQA/env.yml
conda init
eval "$(conda shell.bash hook)"
conda activate nisqa
python --version
pip install transformers
pip install resemblyzer
pip install python-Levenshtein
pip install ipykernel
pip install matplotlib
pip install librosa
pip install torch
pip install pyyaml
pip install pandas
pip install numpy