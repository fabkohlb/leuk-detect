# leuk-detect
Blood cancer (Acute Myeloid Leukemia) prediction model

Single cell leukocyte image classification using CNN

# API Server
Start simple api server command
uvicorn api.simple_api:app --reload

## Set up Access from Virtual Machine to Bucket from a Local Terminal by copying local Access rights
gsutil iam ch serviceAccount:SERVICE_ACCOUNT_OF_VM:roles/storage.admin gs://project_blood_cancer

## Installation
To install the tool, run the following commands:

sudo apt update && sudo apt upgrade -y &&
sudo apt install git && make unzip -y

git clone https://github.com/fabkohlb/leuk-detect.git &&
cd leuk-detect/ &&
cp .env.sample .env

make setup

## Add disk to Google cloud compute VM
List available disks
lsblk
Output for example

Format disk
sudo mkfs.ext4 {Path to disk}

Create mount directory
sudo mkdir /mnt/{any name}

Mount
sudo mount {Path to disk} /mnt/{any name}
