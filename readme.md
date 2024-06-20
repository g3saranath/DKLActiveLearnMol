# CMF Implementation with viDKL

The repository demonstrates the CMF implementation on viDKL with CMF Logging, and Continous Training with a Different Acquisition function to reduce the loss further than the one produced in initial experiments. This helps in logging every experiment and also obtaining the particular checkpoint for continous training/ experiments.

## End-to-End example of Pre-Training -> Continuous Training

1. Install the required packages using the command given below after creating a `venv` or `conda env`.
   > pip install -r requirements.txt
2. Run a CMF server in Port 8080 by following this documentation https://hewlettpackard.github.io/cmf/cmf_server/cmf-server/

3. `DemoP1.ipynb` implements the CMF Logging part of viDKL. It logs 3 stages: Training, Predict and Active Learning.
   50 exploration steps are logging with a chosen acquisition function after running some pre experiment for demonstraion in `Experiment_before_P1.ipynb`, which shows convergence in multiple acquisition functions.
   Hence we take one from them and train for certain steps, and perform continous training in `DemoP2.ipynb`. Once logged push the metadata and artifacts to the cmf-server in port 8080 (code provided in the notebook DemoP1). 
4. The `DemoP2.ipynb` can be run on a different directory also, for simplicity showing in the same directory here. This part pulls the artifacts and metadata, performs some loss analysis based on threshold value, if loss > threshold, then the model at that point will be fetched from cmf artifacts, and continous training will be performed with a different acquisition function.
5. `CMF_query.ipynb` is also provided to show different querying aspects of the mlmd file to obtain the executions and artifacts.
6. The output neo4j graph DB can be accessed from http://localhost:7474 --> Enter username and password for Bolt login as specified in the code.

## For Dynamic Active Learning Experiments:
Please run the file `Representation Learning Experiments-fullset.ipynb` notebook to run experiments of AF, DAF and Normalized DAF for the entire dataset consisting of about 133K samples on Energy Target properties. 
