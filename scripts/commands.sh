#!/bin/bash

# Sample training command
python train.py configs/last_3.yaml --outdir output/voc_person_motor --dump_rpn rpn_output --dump_roi roi_output --load_last

# Test final incremental model (person + motor)
python train.py configs/last_3.yaml --outdir output/voc_person_motor --test_only


