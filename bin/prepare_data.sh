#!/bin/bash

SNLI_ID=1wkAjMu-Pqnm1l-92M7UEp5YEtT1cFgVz
QQP_TRAIN_ID=1dnck-CCIyx8y2xg1vwFzcwXieZJB7ERC
QQP_TEST_ID=1XD-HxzUCTHrzhfvIXOlgqN_MWiiAqM8h

SNLI_DATA=${SNLI_DATA:-https://drive.google.com/uc?export=download&id=${SNLI_ID}}
QQP_DATA_TRAIN=${QQP_DATA_TRAIN:-https://drive.google.com/uc?export=download&id=${QQP_TRAIN_ID}}
QQP_DATA_TEST=${QQP_DATA_TEST:-https://drive.google.com/uc?export=download&id=${QQP_TEST_ID}}


BOT_TRAINLINK=${BOT_TRAINLINK:-https://drive.google.com/file/d/1unxuYa3-6ZrS4W1HLeg0gfqzIYHLVNN1/view?usp=sharing}
BOT_DEVLINK=${BOT_DEVLINK:-https://drive.google.com/file/d/1EtcJmW4qattUEm1RZ8hUfIbCQVViVo53/view?usp=sharing}
GENDER_DEVLINK=${GENDER_DEVLINK:-https://drive.google.com/file/d/1pXy9qD7serdcQzbr9yJgXGytyHf2tflM/view?usp=sharing}
GENDER_TRAINLINK=${GENDER_TRAINLINK:-https://drive.google.com/file/d/1XdDRLCprBPmu-dXZKAzTrl6V2OIcXFoX/view?usp=sharing}

BOT_DIR=Bot
GENDER_DIR=Gender
CORPORA_DIR=corpora

SNLI_DIR=SNLI
QQP_DIR=QQP
CORPORA_DIR=corpora

SNLI_FILE=train_snli.tgz
QQP_FILE_TRAIN=qqp_train.tgz
QQP_FILE_TEST=qqp_test.tgz

BOT_TRAINFILE=dataTrainBot.csv
BOT_DEVFILE=dataDevBot.csv
GENDER_TRAINFILE=dataTrainGender.csv
GENDER_DEVFILE=dataDevGender.csv

mkdir ../${CORPORA_DIR}
cd ../${CORPORA_DIR}
mkdir ${GENDER_DIR} ${BOT_DIR}

function google_drive_big_file_download () {
  CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
  rm -rf /tmp/cookies.txt
}

#wget --no-check-certificate ${SNLI_DATA} -O ${SNLI_DIR}/${SNLI_FILE}
#wget --no-check-certificate ${QQP_DATA_TRAIN} -O ${QQP_DIR}/${QQP_FILE_TRAIN}
#google_drive_big_file_download ${QQP_TEST_ID} ${QQP_DIR}/${QQP_FILE_TEST}

wget ${BOT_TRAINLINK} -O ${BOT_DIR}/${BOT_TRAINFILE}
wget ${BOT_DEVLINK} -O ${BOT_DIR}/${BOT_DEVFILE}
wget ${GENDER_TRAINLINK} -O ${GENDER_DIR}/${GENDER_TRAINFILE}
wget ${GENDER_DEVLINK} -O ${GENDER_DIR}/${GENDER_DEVFILE}





