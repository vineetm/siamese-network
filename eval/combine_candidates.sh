#!/usr/bin/env bash
CDIR="${HOME}/data/ubuntu/ir-candidates/ctx-retr-m0"
SDIR="scores"

mkdir -p $SDIR

python combine_candidates.py -cdir $CDIR -sdir $SDIR