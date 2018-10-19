FROM pytorch/pytorch:0.4_cuda9_cudnn7

ADD . /opt/vgg16
# RUN conda create -n vgg --file /opt/vgg16/requirements.txt
# RUN source activate vgg
# not sure why source activate is not working..
WORKDIR /opt/vgg16
ENTRYPOINT ["python", "main.py"]