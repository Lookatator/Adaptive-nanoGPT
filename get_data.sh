#! /bin/bash

# From https://github.com/facebookresearch/adaptive-span/blob/27b815fee821acce2c2d4104cc0720cb1dc74c37/get_data.sh#L3-L11

mkdir -p data/enwik8
cd data/enwik8
echo "Downloading enwik8 data ..."
wget --continue http://mattmahoney.net/dc/enwik8.zip
wget https://raw.githubusercontent.com/salesforce/awd-lstm-lm/master/data/enwik8/prep_enwik8.py
cd ../..