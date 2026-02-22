# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""PoC constants for v1 scheduler integration."""

# Smaller priority values are scheduled first in PriorityRequestQueue.
# Chat defaults to 0, so PoC must be > 0 to yield under load.
POC_REQUEST_PRIORITY: int = 100
