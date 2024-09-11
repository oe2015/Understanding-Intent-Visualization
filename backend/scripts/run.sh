#!/bin/bash

# Script for executing lambda functions

source "${BASH_SOURCE%/*}/common.sh"

if [ -z "$1" ] ; then
    c_echo $RED "Need to provide the lambda function as the first argument"
    exit 1
fi

cd $(dirname "$1")

# Executing Python script
export PYTHONPATH="$PYTHONPATH:../../cmd"
export PYTHONPATH="$PYTHONPATH:../../cmd/cmn/"
python $(basename "$1")