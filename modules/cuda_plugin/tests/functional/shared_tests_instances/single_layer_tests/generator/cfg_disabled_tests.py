#!/usr/bin/env python3


cfg_disabled_tests = {
    #
    # MatMul: terminate called without an active exception
    'LPCnet-lpcnet_enc:opid24',

    #
    # MaxPool: The end corner is out of bounds at axis 3
    'squeezenet1.1:opid41',
    'squeezenet1.1:opid74',
}
