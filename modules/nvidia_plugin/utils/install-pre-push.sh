#!/bin/bash
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

DIR=$(dirname $0)
cp $DIR/pre-push $DIR/../../../.git/hooks/
