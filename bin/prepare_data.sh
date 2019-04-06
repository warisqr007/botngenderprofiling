#!/bin/bash

SNLI_ID=1wkAjMu-Pqnm1l-92M7UEp5YEtT1cFgVz
QQP_TRAIN_ID=1dnck-CCIyx8y2xg1vwFzcwXieZJB7ERC
QQP_TEST_ID=1XD-HxzUCTHrzhfvIXOlgqN_MWiiAqM8h

BOT_TRAIN_ID=1unxuYa3-6ZrS4W1HLeg0gfqzIYHLVNN1
BOT_DEV_ID=1EtcJmW4qattUEm1RZ8hUfIbCQVViVo53
GENDER_TRAIN_ID=1XdDRLCprBPmu-dXZKAzTrl6V2OIcXFoX
GENDER_DEV_ID=1pXy9qD7serdcQzbr9yJgXGytyHf2tflM


SNLI_DATA=${SNLI_DATA:-https://drive.google.com/uc?export=download&id=${SNLI_ID}}
QQP_DATA_TRAIN=${QQP_DATA_TRAIN:-https://drive.google.com/uc?export=download&id=${QQP_TRAIN_ID}}
QQP_DATA_TEST=${QQP_DATA_TEST:-https://drive.google.com/uc?export=download&id=${QQP_TEST_ID}}


BOT_TRAIN_DATA=${BOT_TRAIN_DATA:-https://drive.google.com/uc?export=download&id=${BOT_TRAIN_ID}}
BOT_DEV_DATA=${BOT_DEV_DATA:-https://drive.google.com/uc?export=download&id=${BOT_DEV_ID}}
GENDER_TRAIN_DATA=${GENDER_TRAIN_DATA:-https://drive.google.com/uc?export=download&id=${GENDER_TRAIN_ID}}
GENDER_DEV_DATA=${GENDER_DEV_DATA:-https://drive.google.com/uc?export=download&id=${GENDER_DEV_ID}}


BOT_DIR=Bot
GENDER_DIR=Gender
CORPORA_DIR=corpora

SNLI_DIR=SNLI
QQP_DIR=QQP
CORPORA_DIR=corpora

SNLI_FILE=train_snli.tgz
QQP_FILE_TRAIN=qqp_train.tgz
QQP_FILE_TEST=qqp_test.tgz

BOT_TRAIN_FILE=dataTrainBot.csv
BOT_DEV_FILE=dataDevBot.csv
GENDER_TRAIN_FILE=dataTrainGender.csv
GENDER_DEV_FILE=dataDevGender.csv

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

wget --no-check-certificate ${BOT_TRAIN_DATA} -O ${BOT_DIR}/${BOT_TRAIN_FILE}
wget --no-check-certificate ${BOT_DEV_DATA} -O ${BOT_DIR}/${BOT_DEV_FILE}
wget --no-check-certificate ${GENDER_TRAIN_DATA} -O ${GENDER_DIR}/${GENDER_TRAIN_FILE}
wget --no-check-certificate ${GENDER_DEV_DATA} -O ${GENDER_DIR}/${GENDER_DEV_FILE}



