#!/bin/bash
# Copyright 2020 ETH Zurich. All Rights Reserved.

set -eu

coverage run  --source=../paramlooper -m unittest discover
coverage report -m
