FROM python:3.10.12-slim

# REPO_DIR contains the path of the repo inside the container, which is 
# expected to be mounted inside and not copied (this is a dev env)
ARG REPO_DIR=./
ARG USERID=1000
ARG GROUPID=1000
ARG DEBIAN_FRONTEND=noninteractive

ENV TZ=Europe/Rome
ENV REPO_DIR=${REPO_DIR}

# Install dependencies
RUN apt-get update -y && apt-get install -y \
    git-flow \ 
    sudo \
    curl \
    gpg \
    graphviz


RUN groupadd -g $GROUPID hrlc
RUN useradd -ms /bin/bash -u $USERID -g $GROUPID hrlcuser
RUN pip install --upgrade pip

# Make sudo easy to use inside the container
RUN echo "hrlcuser ALL=(ALL:ALL) NOPASSWD: ALL" | tee /etc/sudoers.d/hrlcuser

COPY entrypoint.sh /opt/app/entrypoint.sh

ENTRYPOINT [ "bash", "/opt/app/entrypoint.sh" ]
CMD [ "bash" ]
