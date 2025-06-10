#!/bin/bash
# $1: image name
# $2: tag name

  docker volume create \
        --driver local \
        --opt type=cifs \
        --opt device=//192.168.163.214/samba_cci \
        --opt o=addr=192.168.163.214,username=lucazanolo,password=D6X2eZjS4A,file_mode=0555,dir_mode=0555 \
    --name samba_share_folder

docker build --build-arg="USERID=$(id -u)" \
    --build-arg="GROUPID=$(id -g)" \
    --build-arg="REPO_DIR=$(pwd | sed "s/$USER/hrlcuser/")" \
    -t hrlc/$1:$2 .

docker run -it -h $1 \
    -d \
    --name $1_$USER \
    -u $(id -u):$(id -g) \
    -v /home/$USER:/home/hrlcuser \
    -v /media/datapart/lucazanolo:/home/hrlcuser/media \
    -v /mnt/ws_share/gianmarcoperantoni/hrlc_cci_download:/home/hrlcuser/media/hrlc_cci_download \
    -v samba_share_folder:/home/hrlcuser/media/samba_share_folder \
    -w /home/hrlcuser \
    --network host \
    hrlc/$1:$2


