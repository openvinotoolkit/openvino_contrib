// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { awaitAllCallbacks } from '@langchain/core/callbacks/promises';
import { afterAll, jest } from '@jest/globals';

afterAll(awaitAllCallbacks);

// Allow console.log to be disabled in tests
if (process.env.DISABLE_CONSOLE_LOGS === 'true') {
  console.log = jest.fn();
}
