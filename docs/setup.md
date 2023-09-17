# setup docker
- pull `tensorflow v1` docker image: `docker pull <YOUR_DOCKER_IMAGE_NAME>`
- build docker container name `mytf1`, mount your working directory to folder `mrcnn` inside container: `docker run --gpus all --shm-size=30g --name mytf2 -it -v $PWD:/mrcnn <YOUR_DOCKER_IMAGE_NAME> /bin/bash`, example: `docker run --gpus all --shm-size=30g --name mytf2 -it -v $PWD:/mrcnn tensorflow/tensorflow:2.12.0-gpu /bin/bash`

- start docker container by command: `docker start mytf2`

- stop docker container by command: `docker stop mytf2`

- remove docker container by command: `docker rm mytf2`

- remove docker image by command `docker rmi <YOUR_DOCKER_IMAGE_NAME>`

# Install TensorFlow Object Detection API.
- clone source code `tensorflow object-detection` (*You can do this command outside the container*): `git clone https://github.com/tensorflow/models.git`

- go inside docker container by command: `docker exec -it mytf2 /bin/bash`

- check gpu and container connection `nvidia-smi`

- install protobuf compiler: `apt update && apt install protobuf-compiler`

- go to directory `research` directory: `cd models/research`

- compile protoc file to python file: `protoc object_detection/protos/*.proto --python_out=.`

- copy setup file to outside: `cp object_detection/packages/tf2/setup.py .`

- install `object-detection` by `setup.py` file: `python3 -m pip install .`

- test your installation: `python3 object_detection/builders/model_builder_tf2_test.py`
