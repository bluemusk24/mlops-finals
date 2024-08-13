### 1. Project Paper: [UCI Land Mine](https://github.com/bluemusk24/mlops-final-project/blob/main/UCI_Mine_Paper/Article.pdf)

### Project definition : Passive Land Mines Detection with Hybrid Machine-Learning Models:
Land mines can be detected with high precision active detectors. However, the principle of active detectors involves sending electrical signals to the environment which could trigger mine blasting, cause mine explosion and lead to dreadful dangers. To prevent this deadly occurence from active detectors, passive land mine detectors were introduced and used. Passive mine detectors, though less dangerous as active detectors, have some limitations. In this project, we will eliminate the handicaps of passive mine detectors with machine learning techniques. The [Kaggle dataset](https://www.kaggle.com/datasets/spikey19/uci-land-mines-dataset) for this project is measured under different conditions(features) to classify different types of mines detected.

The [land-mine dataset](https://github.com/bluemusk24/mlops-final-project/blob/main/01-notebook/Mine_Dataset.csv) variables are described below:

V = output voltage of sensor due to magnetic distortion

H = height of the sensor from the ground

S = 6 different soil types depending on moisture condition: 0.0-(Dry & Sandy), 0.2-(Dry & Humus), 0.4-(Dry & Limy), 0.6-(Humid & Sandy), 0.8-(Humid & Humus), 1.0-(Humid & Limy).

M = 5 different land-mine types: 0(Null), 1(Anti-Tank), 2(Anti-Personnel), 3(Booby Trapped Anti-Personnel), 4(M14 Anti-Personnel)


### 2. Environment Setup on Local Machine Terminal.

* download and install [anaconda](https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh) in my base environment

```bash
wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh

bash Anaconda3-2023.09-0-Linux-x86_64.sh
```

* view the contents of installed anaconda.

```bash
cat ~/.bashrc
```

* view the path of python from installed anaconda and get the latest installed version.

```bash
which python

python -V

python --version
```

* ensure Docker-desktop is installed locally and running on machine. run these containers to test docker locally:

```bash
docker run -it ubuntu bash

docker run hello-world
```

##### Note: same procedures bar downloading and installing Docker applies to Github codespaces - if used locally. This is because Docker comes pre-installed on Github codespaces.

* create a directory and a conda virtual environment for the final project

```bash
mkdir mlops-final-project

conda create -n project-env python=3.11

conda activate project-env
```

* see created conda virtual environment and activate the required one

```bash
conda env list

conda activate project-env
```

* install neccesary requirements in the virtual environment

```bash
pip install -r requirements.txt
```


### 3. Data Preparation and training the best ML model notebook:

* create folder notebook

```bash
mkdir 01-notebook
```

* The Cross-Industry Standard Process for Data Mining (CRISP-DM) is located in the jupyter notebook [land_mine_detection](https://github.com/bluemusk24/mlops-final-project/blob/main/01-notebook/land_mine_detection.ipynb).

* All the trained and hyperparameter-tuned ML models can be found in the folder [models](https://github.com/bluemusk24/mlops-final-project/tree/main/01-notebook/models). Support Vector Machine is the best-trained ML model, which will be used in this project.

* The jupyter notebook [landmine_train](https://github.com/bluemusk24/mlops-final-project/blob/main/03-model-training/landmine_train.ipynb) contains codes procedures that will be used to create a model training pipeline.


### 4. Experiment Tracking with MLflow:

* create folder experiment-tracking

```bash
mkdir 02-experiment-tracking
```

* get the root directory and install mlflow in the working directory mlops-final-project

```bash
pwd

pip install mlflow
```

* copy the jupyter notebook [land_mine_detection](https://github.com/bluemusk24/mlops-final-project/blob/main/01-notebook/land_mine_detection.ipynb) and dataset [dataset](https://github.com/bluemusk24/mlops-final-project/blob/main/01-notebook/Mine_Dataset.csv)  into this directory

```bash
cp 01-notebook/land_mine_detection.ipynb 02-experiment-tracking/

cp 01-notebook/Mine_Dataset.csv 02-experiment-tracking/
```

* start mlflow server with sqlite as backend database to store artifact and metadata locally

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db

mlflow server --backend-store-uri sqlite:///backend.db --default-artifact-root ./artifacts_local
```

* the notebook for tracking experiment is [land_mine_experiment.ipynb](https://github.com/bluemusk24/mlops-final-project/blob/main/02-experiment-tracking/land_mine_experiment.ipynb). Models with the best accuracy and roc_auc_score were registered in the model registry on MLflow.

* the notebook [stage-production-archive.ipynb](https://github.com/bluemusk24/mlops-final-project/blob/main/02-experiment-tracking/stage-prodution-archive.ipynb) in experiment-tracking folder indicates which model is pushed to production, staging and archived.

* [Mlflow_tracked_experiment screenshot](https://github.com/bluemusk24/mlops-final-project/blob/main/02-experiment-tracking/Screenshot%20(36).png)


### 5. Model Training Pipeline

* create folder model-training

```bash
mkdir 03-model-training
```

* turn the notebook into a python script

```bash
jupyter nbconvert --to script landmine_train.ipynb
```

* the model training pipeline can be found in this script [landmine_train.py](https://github.com/bluemusk24/mlops-final-project/blob/main/03-model-training/landmine_train.py)


### 6. Deployment

* create folder deployment and web-service inside deployment folder

```bash
mdkir 04-deployment

mkdir web-service
```

* get the installed version of scikit-learn and install with pipenv

```bash
pip freeze | grep scikit-learn

pipenv install scikit-learn==' '
```

* create a [predict.py](https://github.com/bluemusk24/mlops-final-project/blob/main/04-deployment/web-service/predict.py) web-service file to load model for predictions

```bash
python3 predict.py
```

* create a [predict-test](https://github.com/bluemusk24/mlops-final-project/blob/main/04-deployment/web-service/predict-test.py) to test to the app-predict and a [test](https://github.com/bluemusk24/mlops-final-project/blob/main/04-deployment/web-service/test.py) script that uses request from the flask app

```bash
python predict-test.py

python test.py
```

* install gunicorn, run the predict app with gunicorn and test script

```bash
pipenv install gunicorn

pipenv run gunicorn --bind localhost:9696 predict:app

python3 test.py
```

* develop a dockerfile below, build the image, run the container and run the test.py script

```bash
FROM python:3.10.12-slim
RUN pip install -U pip
RUN pip install pipenv
WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", "./"]
RUN pipenv install --system --deploy
COPY ["predict.py", "model_SVC.bin", "./"]
EXPOSE 9696
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app]

docker build -t landmine:v1 .

docker run -it --rm -p 9696:9696 landmine:v1

python3 test.py
```

#### Deploying the containerized landmine model in a Kubernetes cluster locally with kind

* download kind locally, make kind executable and list to view.

```bash
wget https://kind.sigs.k8s.io/dl/v0.20.0/kind-linux-amd64 -O kind

chmod +x ./kind

ls
```

* create a kubernetes cluster locally with  kind and node image (kindest/node:v1.27.3)

```bash
kind create cluster
```

* configure kubectl to kind, list all services, deployment and pods running in this Kubernetes cluster.

```bash
kubectl cluster-info --context kind-kind

kubectl get service

kubectl get pod

kubectl get deployment
```

* build the docker image:tag and run the container

```bash
docker build -t landmine:v1 .

docker run -it --rm -p 9696:9696 landmine:v1
```

* load docker image into our cluster. wait till it brings out the working directory

```bash
kind load docker-image landmine:v1
```

* go to extensions of vsc, type and install Kubernetes from Microsoft.

* create a [model-deployment.yaml](https://github.com/bluemusk24/mlops-final-project/blob/main/04-deployment/web-service/model-deployment.yaml) file in the working directory and copy these codes into the file. Type deployment and select the Kubernetes deployment that pops up.

```bash
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-model
spec:
  replicas: 1
  selector:
    matchLabels:
      app: my-model
  template:
    metadata:
      labels:
        app: my-model
    spec:
      containers:
      - name: my-model
        image: landmine:v1
        resources:
          limits:
            memory: "128Mi"
            cpu: "500m"
        ports:
        - containerPort: 9696
```

* load model-deployment file into the cluster; get deployment and pod.

```bash
kubectl apply -f model-deployment.yaml

kubectl get deployment

kubectl get pod
```

* get pod description. Press q to quit, and htop to confirm CPU usage

```bash
kubectl describe pod landmine-model-9559b77bd-grdv9

kubectl describe pod landmine-model-9559b77bd-grdv9 | less

q

htop
```

* test the created deployment (pods) by forwarding the port from the host machine to deployment container.

```bash
kubectl port-forward landmine-model-9559b77bd-grdv9 9696:9696 (OR use 9797:9696)
```

* run in a new termnial and ctrl c to stop.

```bash
python3 test.py
```

* create a [model-service.yaml](https://github.com/bluemusk24/mlops-final-project/blob/main/04-deployment/web-service/model-service.yaml) file in the working directory and copy these codes into the file. Type service and select the Kubernetes service yaml file that pops up. Edit the yaml file below:

```bash
apiVersion: v1
kind: Service
metadata:
  name: my-model
spec:
  type: LoadBalancer
  selector:
    app: my-model
  ports:
  - port: 80
    targetPort: 9696
```

* load model-service file into the cluster; get the services

```bash
kubectl apply -f model-service.yaml

kubectl get service
kubectl get svc
```

* test the created service by forwarding the port from the host machine to cluster. You get a port forward response

```bash
kubectl port-forward service/my-model 9696:80 (OR use 9797:80)
```

* run in a new termnial and ctrl c to stop.

```bash
python3 test.py
```

#### DEPLOYING ALL THE YAML FILES INTO AWS ELASTIC KUBERNETES SERVICE USING EKSCTL: a CLI tool for creating Amazon EKS clusters.

* run these codes to download eksctl, extract eksctl, remove the tag archive and move eksctl to local bin.

```bash
wget https://github.com/eksctl-io/eksctl/releases/latest/download/eksctl_$(uname -s)_amd64.tar.gz

ls

tar xzfv eksctl_Linux_amd64.tar.gz

rm eksctl_Linux_amd64.tar.gz

sudo mv /tmp/eksctl /usr/local/bin
```
* Link of [eksctl installation](https://eksctl.io/installation/)

* create an [eks-config.yaml](https://github.com/bluemusk24/mlops-final-project/blob/main/04-deployment/web-service/eks-config.yaml) file in the working and paste the codes below in it

```bash
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig

metadata:
  name: mlops-eks
  region: us-east-1

nodeGroups:
  - name: ng-1
    instanceType: m5.large
    desiredCapacity: 10
  - name: ng-m5-xlarge
    instanceType: m5.xlarge
    desiredCapacity: 1
```

* configure AWS, create cluster. wait for it to be executed and perform other proecesses.

```bash
aws configure

eksctl create cluster -f eks-config.yaml
```

* create an ECR repository and execute the commands below. Also, login to ECR and push the model

```bash
aws ecr create-repository --repository-name mlops-ecr

ACCOUNT=**********
REGION=us-east-1
REGISTRY=mlops-ecr
PREFIX=${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${REGISTRY}
MODEL_LOCAL=landmine:v1
MODEL_REMOTE=${PREFIX}:landmine-test
docker tag ${MODEL_LOCAL} ${MODEL_REMOTE}

$(aws ecr get-login --no-include-email)

docker push ${MODEL_REMOTE}
```

* Get outcome URI of model to include in the model deployment file.

```bash
echo ${MODEL_REMOTE}
```

* check if the cluster is completely created in the other terminal - it should show the working directory. Run these codes

```bash
kubectl get nodes

docker ps

kubectl apply -f  model-deployment.yaml

kubectl apply -f  model-service.yaml

kubectl get pod

kubectl get service(or svc)

kubectl port-forward my-model-6454694896-wlczs('pod name') 9797:9696

kubectl port-forward service/my-model 9797:80
```

* Using telnet and DNS name (url) from Load balancer. Paste the url in (test.py)[] script.

```bash
telnet a8b6fae0108944af5aa148677d8d3a2d-905019242.us-east-1.elb.amazonaws.com 80

python3 test.py
```

* login to AWS, checkout created EKS to view the created cluster, EC2 instances, load Balancer. Always restrict access from public usage. Ask experts in your company for that.

[Amazon_EKS_1](https://github.com/bluemusk24/mlops-final-project/blob/main/04-deployment/web-service/Screenshot%20(39).png)
[Amazon_EKS_2](https://github.com/bluemusk24/mlops-final-project/blob/main/04-deployment/web-service/Screenshot%20(40).png)
[Amazon_EKS_3](https://github.com/bluemusk24/mlops-final-project/blob/main/04-deployment/web-service/Screenshot%20(41).png)

*  delete the cluster to avoid been charged. It deletes EC2 instances and load balancer as well. cluster name is located in eks-config.yaml file.

```bash
eksctl delete cluster --name mlops-eks
```

#### In this section, we connect the flask app (predict.py) to our model registry on MLflow.

* create a folder web-service-mlflow and copy predict.py, test.py, Pipfile and Pipfile.lock into the folder

```bash
mkdir web-service-mlflow
```

* install the following libraries and update requirements.txt file

```bash
pipenv install boto3

pipenv install awscli

pipenv install botocore
```

* create a s3 bucket 'mlops-remotebucket' on AWS to be the default artifact store for the model registry.

* run this code to start MLflow with sqlite as database and S3 bucket as artifact store.

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root s3://'mlops-remote-bucket'/
```

* create and run a no-pipeline notebook [nopipeline-mlflow](https://github.com/bluemusk24/mlops-final-project/blob/main/04-deployment/web-service-mlflow/nopipeline-mlflow.ipynb) to track experiment in MLflow. This notebook includes both dictvectorizer and the model. connect the [predict1.py_flaskapp](https://github.com/bluemusk24/mlops-final-project/blob/main/04-deployment/web-service-mlflow/predict1.py) to the nopipeline model from mlflow by inserting the new run_id. Run the codes below afterwards

```bash
python3 predict1.py

python3 test.py
```

* create and run a pipeline notebook [pipeline-mlflow](https://github.com/bluemusk24/mlops-final-project/blob/main/04-deployment/web-service-mlflow/pipeline-mlflow.ipynb) to track experiment in MLflow. This notebook combibes both dictvectorizer and the model as a pipeline. connect the [predict2.py_flaskapp](https://github.com/bluemusk24/mlops-final-project/blob/main/04-deployment/web-service-mlflow/predict2.py) to the pipeline model from mlflow by inserting the new run_id. Run the codes below afterwards

```bash
python3 predict2.py

python3 test.py
```

[aws_pic1](https://github.com/bluemusk24/mlops-final-project/blob/main/04-deployment/web-service-mlflow/pic1_aws.png)
[aws_pic2](https://github.com/bluemusk24/mlops-final-project/blob/main/04-deployment/web-service-mlflow/pic2_aws.png)
[mlflow_pic1](https://github.com/bluemusk24/mlops-final-project/blob/main/04-deployment/web-service-mlflow/pic1_mlflow.png)
[mlflow_pic2](https://github.com/bluemusk24/mlops-final-project/blob/main/04-deployment/web-service-mlflow/pic2_mlflow.png)
[mlflow_pic3](https://github.com/bluemusk24/mlops-final-project/blob/main/04-deployment/web-service-mlflow/pic3_mlflow.png)
[mlflow_pic4](https://github.com/bluemusk24/mlops-final-project/blob/main/04-deployment/web-service-mlflow/pic4_mlflow.png)

#### Batch Deployment

* create a folder batch-deployment in the directory 04-deployment. copy codes to  the [batch-processing.ipynb](https://github.com/bluemusk24/mlops-final-project/blob/main/04-deployment/batch-deployment/batch-processing.ipynb) notebook.

```bash
mkdir batch-deployment

cp ../web-service-mlflow/pipeline-mlflow.ipynb batch-processing.ipynb
```

* convert the [batch-processing.ipynb](https://github.com/bluemusk24/mlops-final-project/blob/main/04-deployment/batch-deployment/batch-processing.ipynb) notebook to a python [batch-processing.py](https://github.com/bluemusk24/mlops-final-project/blob/main/04-deployment/batch-deployment/batch-processing.py) script. Run the pipeline script

```bash
jupyter nbconvert --to script batch-processing.ipynb batch-processing.py

python3 batch-processing.py
```


### 7. Pipeline Orchestration with MAGE

* create folder mage-pipeline-orchestration and copy the Mage folder (mlops) to it.

```bash
mdkir 05-mage-pipeline-orhestration

cd mlops
```

* Launch Mage in a terminal and open the Mage UI with the link below:

```bash
./scripts/start.sh

http://localhost:6789
```
* screenshots of ML pipeline orchestration.
[Mage1](https://github.com/bluemusk24/mlops-final-project/blob/main/05-mage-pipeline-orchestration/mage1.png)
[Mage2](https://github.com/bluemusk24/mlops-final-project/blob/main/05-mage-pipeline-orchestration/mage2.png)
[Mage3](https://github.com/bluemusk24/mlops-final-project/blob/main/05-mage-pipeline-orchestration/mage3.png)
[Mage4](https://github.com/bluemusk24/mlops-final-project/blob/main/05-mage-pipeline-orchestration/mage4.png)
[Mage5](https://github.com/bluemusk24/mlops-final-project/blob/main/05-mage-pipeline-orchestration/mage5.png)
[Mage6](https://github.com/bluemusk24/mlops-final-project/blob/main/05-mage-pipeline-orchestration/mage6.png)
[Mage7](https://github.com/bluemusk24/mlops-final-project/blob/main/05-mage-pipeline-orchestration/mage7.png)
[Mage8](https://github.com/bluemusk24/mlops-final-project/blob/main/05-mage-pipeline-orchestration/mage8.png)
[Mage9](https://github.com/bluemusk24/mlops-final-project/blob/main/05-mage-pipeline-orchestration/mage9.png)
[Mage10](https://github.com/bluemusk24/mlops-final-project/blob/main/05-mage-pipeline-orchestration/mage10.png)
[Mage11](https://github.com/bluemusk24/mlops-final-project/blob/main/05-mage-pipeline-orchestration/mage11.png)
[Mage12](https://github.com/bluemusk24/mlops-final-project/blob/main/05-mage-pipeline-orchestration/mage12.png)
[Mage13](https://github.com/bluemusk24/mlops-final-project/blob/main/05-mage-pipeline-orchestration/mage13.png)
[Mage14](https://github.com/bluemusk24/mlops-final-project/blob/main/05-mage-pipeline-orchestration/mage14.png)
[Mage15](https://github.com/bluemusk24/mlops-final-project/blob/main/05-mage-pipeline-orchestration/mage15.png)


### 8. Monitoring

* create folder monitoring and cd to it. Updata the requirements.txt file by installing the following with pip:

```bash
mkdir 06-monitoring

pip install prefect tqdm requests joblib pyarrow psycopg psycopg_binary evidently
```

#### Monitoring with Evidently

* Monitor the prediction column and data drift of reference data, current data and additional data with Evidently report. check [landmine_monitoring.ipynb](https://github.com/bluemusk24/mlops-final-project/blob/main/06-monitoring/landmine_monitoring.ipynb) for codes and processes. run these codes to start/view the Evidently dashboard:

```bash
evidently ui --help

evidently ui

curl http://0.0.0.0:8000
```

* open a browser and type 'localhost:8000'. This opens evidently user interface with the project name. Stop and Start evidently ui to view updated projects. A folder workspace - contains evidently metadata - should be created in the working directory.

#### Loading the data into Postgres/Adminer and Monitoring with Grafana

* create a [docker-compose.yaml](https://github.com/bluemusk24/mlops-final-project/blob/main/06-monitoring/docker-compose.yaml) file to run multiple containers (Grafana, PostgreSql, adminer-to manage the PostgreSQL database)

```bash
version: '3.7'

volumes:
  grafana_data: {}

networks:
  front-tier:
  back-tier:

services:
  db:
    image: postgres
    restart: always
    environment:
      POSTGRES_PASSWORD: example
    ports:
      - "5432:5432"
    networks:
      - back-tier

  adminer:
    image: adminer
    restart: always
    ports:
      - "8080:8080"
    networks:
      - back-tier
      - front-tier

  grafana:
    image: grafana/grafana
    user: "472"
    ports:
      - "3000:3000"
    volumes:
      - ./config/grafana_datasources.yaml:/etc/grafana/provisioning/datasources/datasource.yaml:ro
      - ./config/grafana_dashboards.yaml:/etc/grafana/provisioning/dashboards/dashboards.yaml:ro
      #- ./dashboards:/opt/grafana/dashboards
    networks:
      - back-tier
      - front-tier
    restart: always
```

* create a config folder in the monitoring directory to store grafana_datasources.yaml file and run this to build the containers:

```bash
docker-compose up --build
```

* access each container through their specified ports in the docker-compose yaml file. eg localhost:3000 for grafana, localhost:8080 for adminer, localhost:5432 for PostgreSQL- the PostgreSQL is inside the Grafana.

* the python script [landmine_evidently.py](https://github.com/bluemusk24/mlops-final-project/blob/main/06-monitoring/landmine_evidently.py) contains code to create a Postgres database, infuse the data into the database, access the database via adminer and monitor the metrics on Grafana. Run this code to send data to adminer/Postgres and monitored on Grafana

```bash
python3 landmine_evidently.py
```

[Evidently_pic1](https://github.com/bluemusk24/mlops-final-project/blob/main/06-monitoring/pic_evidently1.png)
[Evidently_pic2](https://github.com/bluemusk24/mlops-final-project/blob/main/06-monitoring/pic_evidently2.png)
[Grafana_pic](https://github.com/bluemusk24/mlops-final-project/blob/main/06-monitoring/pic_grafana.png)


### 9. Best Practices

#### Unit test with pytest

* create directory best practices and copy the web-service code from deployment. Name as code and open on vsc.

```bash
mkdir unit_test

cp -r 04-deployment/web-service unit_test

code .
```

* run codes below to install pipenv, pytest(for local test), create a test folder, get actual pipenv and select a python interpreter on vs code.

```bash
pipenv install

mkdir test

pipenv install --dev pytest

pipenv --venv

ctrl shift p
```

* check codes in script [model_test.py](https://github.com/bluemusk24/mlops-final-project/blob/main/unit_test/model_test.py) for unit test of the function. Run Testing icon on VScode or by CLI:

```bash
pipenv run pytest unit_test/
```
* Note: the above output is similar to running the predict.py flask app and predict-test.py

#### Integration Test with Docker

* install Deep Diff to tells us where the assertion error is to compare dictionaries outcomes.

```bash
mkdir integration_test

pipenv install --dev deepdiff

pipenv shell
```

* build the updated dockerfile image to include installed diff from the pipfile. run python3 test_docker.py.

```bash
FROM python:3.10.12-slim
RUN pip install -U pip
RUN pip install pipenv
WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", "./"]
RUN pipenv install --system --deploy
COPY ["predict.py", "model_SVC.bin", "./"]
EXPOSE 9696
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]

docker build -t landmine:v1 .

docker run -it --rm -p 9696:9696 landmine:v1

python3 test_docker.py
```

* automate the whole process in the integration test: create a script [run.sh](https://github.com/bluemusk24/mlops-final-project/blob/main/integration_test/run.sh) and [docker-compose.yaml](https://github.com/bluemusk24/mlops-final-project/blob/main/integration_test/docker-compose.yaml) file in the integration_test folder. Run the run.sh script depending on which directory. Also,  check for error code after running ./run.sh script: 0=no error, non-zero=error - this will give some logs of the error encountered.

```bash
integration_test/./run.sh

echo $?
```

#### Code quality tools: Linting and Formatting

* install pylint in its dev dependency and not for production. see the update pipfile for installed pylint, version of installed pylint and see the code line errors to fix and ratings in the working directory.

```bash
pipenv install --dev pylint

which pylint

pylint --recursive=y .
```

* pyproject.toml can be used instead of pylint to keep configuration that will be ignored or suppress in our codes. check the [pyproject.toml](https://github.com/bluemusk24/mlops-final-project/blob/main/pyproject.toml) file to fix errors before running a CI/CD job and pushing project to git.


```bash
pylint --recursive=y .

echo $?
```

* install black and isort for formatting and sorting codes.

```bash
pipenv install --dev black isort black[jupyter]
```

* black formatting: run to see scripts to be formatted and unchanged, apply reformatting and see ratings of codes.

```bash
black --diff .

black --skip-string-normalization --diff .

black .

pylint --recursive=y .
```

* isort code formatting, application and see ratings.

```bash

isort --diff .

isort .

pylint --recursive=y .
```

* run all the formatting processes and test in a single code
```bash
isort .
black .
pylint --recursive=y .
pytest unit_test/
```

#### Git pre-commit hooks:

* these are necessary processes before committing to git. The python tool for this is pre-commit. Run these codes accordingly:

```bash
pipenv install --dev pre-commit
git init
ls -a
pre-commit
pre-commit help
pre-commit sample-config
pre-commit sample-config > .pre-commit-config.yaml
cat .pre-commit-config.yaml
ls .git/hooks
less .git/hooks/pre-commit
```

*  view the docs installed in pre-commit. if working in a team, they must run pre-commit install to have access to the hooks. this is bcos .git is not viewed in git. Also place pycache in the gitignore file so it;s not commited to git.

```bash
pre-commit install
git status
touch .gitignore
git status
git add .
git commit -m 'initial commit'
git diff
git add .
git commit -m 'fixes from pre-commit default hooks'
git status
```

* just added the hooks in the pre-commit.yaml file. we can add other built-in hooks in the yaml file. google pre-commit.com/hooks to see what can be added to the pre-commit.yaml file. In our case, we include isort, black, pylint, pytest. check the [.pre-commit-config.yaml](https://github.com/bluemusk24/mlops-final-project/blob/main/.pre-commit-config.yaml) file for codes

```bash
pre-commit autoupdate
git status
git add .
git commit -m 'update-commit'
```
* git status, git init(create a git repo), ls -a, pre-commit install(install pre-commit hooks), git status, git add . , git commit -m 'initial commit', rm -rf git (this removes .git repo bcos it's not the root directory).

#### Makefiles and make: 

* Makefiles and make run a process incase you forget the codes for that process. Also, some commands depend on the preceding commands to run their processes. create a [makefile](https://github.com/bluemusk24/mlops-final-project/blob/main/Makefile) for codes and scripts folder for [publish](https://github.com/bluemusk24/mlops-final-project/blob/main/scripts/publish.sh).

```bash
make --version
make publish
```

### Technology Used

* MLflow
* AWS S3 bucket and Elastic Kubernetes Service
* Grafana
* Evidently
* Mage
* Docker
* Kind-Kubernetes



[mlops_pics](<https://github.com/bluemusk24/mlops-final-project/blob/main/mlops%20project%20picx.png>)