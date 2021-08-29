# Final Project
This repository contains:
- Training and testing code for a VAE-based speaker-embedding generator compatible with AutoVC
- VCEvalKit: a non-intrusive voice conversion evaluation toolkit

## Speaker-Embedding Generator
The speaker embedding generator is a module that can be used in combination with AutoVC for converting speech towards non-existing ficticious voices. Moreover, it is possible to perform arbitrarily alter the identity characteristics of a speaker, or to combine several speakers voices to make a new one.

For training and testing, checkout the speakersVAE.ipynb in this repository.

## VCEvalKit
To use VCEvalKit, the module must be downloaded and all dependencies must be installed on the system. This can be done by executing the following commands one after another:

```
git clone https://github.com/antovespoli2/finalProject

cd finalProject/VCEvalKit

chmod +x install.sh

./install.sh
```


Once the system has been installed, it can be used by running the following command with the associated options:

```
  python main.py [options] <path/to/metadata_file>
  
  options: 
    -h 			Show help message

    -o <file>		To generate a report with a given filename

    -m <metric>		To compute a single evaluation metri, options are “content”, “style”, 
                        or “neural_mos”

    -v			To operate in verbose mode
```
  
Here, the ``` metadata_file ``` is a file that must be created from the user and that is used by the program to locate in the filesystem and match together the associated original audios, target speaker’s audios, and generated audios. The structure of the file is a list of dictionaries in the usual Python formatting where each dictionary represents one complete set of files. The sample structure of the metadata file is shown below:
  
```
  [ 
    { “original_audio”  :  <path/to/original_audio1>,
      “target_speaker”  :  <path/to/target_speaker1>,
      “generated_audio” :  <path/to/generated_audio1> },
      
    …
      
    { “original_audio”  :  <path/to/original_audioN>,
      “target_speaker”  :  <path/to/target_speakerN>,
      “generated_audio” :  <path/to/generated_audioN>} 
  ]
```
  

## References
Besides the code provided here, my project adopted various publicly available implementations and pre-trained models.

#### Text-based method
Automatic Speech Recognition (ASR) system: https://huggingface.co/facebook/wav2vec2-large-960h-lv60-self \
Speaker conditional Tacotron: https://github.com/CorentinJ/Real-Time-Voice-Cloning

#### Spectrogram morphing methods
StarGAN-VC: https://github.com/liusongxiang/StarGAN-Voice-Conversion \
AutoVC: https://github.com/auspicious3000/autovc \
HiFi-GAN vocoder: https://github.com/jik876/hifi-gan 

#### VCEvalKit
Automatic Speech Recognition (ASR) system: https://huggingface.co/facebook/wav2vec2-large-960h-lv60-self \
Speaker encoder: https://github.com/resemble-ai/Resemblyzer \
Neural MOS predictor: https://github.com/sky1456723/Pytorch-MBNet
