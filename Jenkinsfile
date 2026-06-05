#!groovy
// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


properties([
    parameters([
        booleanParam(defaultValue: false,
                     description: 'Cancel the rest of parallel stages if one of them fails and return status immediately',
                     name: 'failFast'),
        booleanParam(defaultValue: true,
                     description: 'Whether to propagate commit status to GitHub',
                     name: 'propagateStatus'),
        string(defaultValue: '',
               description: 'Pipeline shared library version (branch/tag/commit). Determined automatically if empty',
               name: 'library_version')
    ])
])
loadOpenVinoLibrary {
    entrypoint(this)
}
