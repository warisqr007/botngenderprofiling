#!/bin/bash

SNLI_ID=1wkAjMu-Pqnm1l-92M7UEp5YEtT1cFgVz
QQP_TRAIN_ID=1dnck-CCIyx8y2xg1vwFzcwXieZJB7ERC
QQP_TEST_ID=1XD-HxzUCTHrzhfvIXOlgqN_MWiiAqM8h

SNLI_DATA=${SNLI_DATA:-https://drive.google.com/uc?export=download&id=${SNLI_ID}}
QQP_DATA_TRAIN=${QQP_DATA_TRAIN:-https://drive.google.com/uc?export=download&id=${QQP_TRAIN_ID}}
QQP_DATA_TEST=${QQP_DATA_TEST:-https://drive.google.com/uc?export=download&id=${QQP_TEST_ID}}


Bot_trainlink =${Bot_trainlink:-https://drive.google.com/file/d/1unxuYa3-6ZrS4W1HLeg0gfqzIYHLVNN1/view?usp=sharing}
Bot_devlink = ${Bot_devlink:-https://drive.google.com/file/d/1EtcJmW4qattUEm1RZ8hUfIbCQVViVo53/view?usp=sharing}
Gender_devlink = ${Gender_devlink:-https://drive.google.com/file/d/1pXy9qD7serdcQzbr9yJgXGytyHf2tflM/view?usp=sharing}
Gender_trainlink = ${Gender_trainlink:-https://drive.google.com/file/d/1XdDRLCprBPmu-dXZKAzTrl6V2OIcXFoX/view?usp=sharing}

Bot_DIR = Bot
Gender_DIR = Gender
CORPORA_DIR=corpora

SNLI_DIR=SNLI
QQP_DIR=QQP
CORPORA_DIR=corpora

SNLI_FILE=train_snli.tgz
QQP_FILE_TRAIN=qqp_train.tgz
QQP_FILE_TEST=qqp_test.tgz

Bot_trainfile = dataTrainBot.csv
Bot_devfile = dataDevBot.csv
Gender_trainfile = dataTrainGender.csv
Gender_devfile = dataDevGender.csv

mkdir ../${CORPORA_DIR}
cd ../${CORPORA_DIR}
mkdir ${Gender_DIR} ${Bot_DIR}

function google_drive_big_file_download () {
  CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
  rm -rf /tmp/cookies.txt
}

#wget --no-check-certificate ${SNLI_DATA} -O ${SNLI_DIR}/${SNLI_FILE}
#wget --no-check-certificate ${QQP_DATA_TRAIN} -O ${QQP_DIR}/${QQP_FILE_TRAIN}
#google_drive_big_file_download ${QQP_TEST_ID} ${QQP_DIR}/${QQP_FILE_TEST}

wget ${Bot_trainlink} -O ${Bot_DIR}/${Bot_trainfile}
wget ${Bot_devlink} -O ${Bot_DIR}/${Bot_devfile}
wget ${Gender_trainlink} -O ${Gender_DIR}/${Gender_trainfile}
wget ${Gender_devlink} -O ${Gender_DIR}/${Gender_trainfile}





