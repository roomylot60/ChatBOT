FROM ufoym/deepo:cpu

VOLUME ["/$(pwd)/data"]
VOLUME ["/$(pwd)/config"]
ENTRYPOINT "/bin/bash"

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y openjdk-8-jdk wget curl git language-pack-ko

RUN locale-gen en_US.UTF-8 && \
    update-locale LANG=en_US.UTF-8

RUN pip install --update && \
    pip install -y --upgrade && \
    pip install -y konlpy wordcloud Pillow