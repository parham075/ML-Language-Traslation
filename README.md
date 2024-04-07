# Machine Learning Language Translation from English To Persian

For creating the environment and executing the Jupyter Notebook, open a new terminal and execute the following commands:
```python
conda env create -f environment.yml
conda activate trnsalate
python -m ipykernel install --user --name "trnsalate"
```




# Execute a training job on minikube cluster

 
To start a training job on a remote machine please create your docker images locally from [train-translator](#) .

## train-translator: 
In this directory the user can creating a docker image locally which is responsible for training a sequence to sequence model used for translating a scentence from English to Farsi.

> For more information about how to create the docker image please follow the instructions in [README.md](train-translator/README.md).


## app-package
After creating your docker images, you can execute your training job using `calrissian`(an executor for [cwl](https://www.commonwl.org/) files) on a minikube cluster from this directory. 

