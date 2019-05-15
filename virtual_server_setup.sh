#!/bin/sh
yes | ssh-keygen -f ~/.ssh/id_rsa -t rsa -b 4096 -P ""
echo "Copying ssh key to server (use server password)"
ssh-copy-id mario@portalgatan.mynetgear.com
sudo apt update
# sudo apt upgrade -y
sudo apt install ffmpeg python3-pip
pip3 install -r requirements.txt
