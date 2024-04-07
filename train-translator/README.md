# `train-translator` module:
This directoy helps the user to create `train-translator` module which responsible for training a sequence to sequence model used for translating a scentence from English to Farsi. 

**Inputs**:
>- DATA_PATH: path to data
>- NUM_SAMPLES: NUM_SAMPLES
>- BATCH_SIZE: BATCH_SIZE
>- EPOCHS: EPOCHS
**Output**: 
>- trained model: A tensorflow model in `.tf` format. 
>- training history: A file to save history of training in `.csv` format.

> Notice: The application package will generate a list of directories containing a single output's files.


## Containerize `train-translator` module on user's Local PC:
The user is able to build a docker image to containerized the module using commands bellow(`not recommended`). However this image is already built and accessable on a remote github container registery which eliminate the user's need to create an image locally.
  ```python
  minikube start --driver=docker
  minikube image build -t ghcr.io/parham075/train_translator:latest .
  ```
> Notice: Make sure the Docker, and minikube installled on user's local PC.