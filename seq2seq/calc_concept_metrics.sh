#!/usr/bin/env bash
GT=$1
PREDS=$2

python calc_concept_metrics.py -gt $GT -preds $PREDS