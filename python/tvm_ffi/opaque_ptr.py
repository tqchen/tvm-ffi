# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Opaque-pointer conversion hooks for third-party Python classes.

Populate :data:`handlers` during application setup, before instances of the
registered classes are passed to TVM FFI.  Treat the mapping as read-only once
FFI calls begin.  Dispatch uses exact classes rather than subclass matching.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any


handlers: dict[type, Callable[[Any], int]] = {}
"""Map exact Python classes to callables returning their opaque pointer value."""


if os.environ.get("TVM_FFI_BUILD_DOCS", "0") == "0":
    try:
        import torch
    except ImportError:
        pass
    else:
        _event_cls = getattr(torch, "Event", None)
        if isinstance(_event_cls, type) and hasattr(_event_cls, "event_id"):
            handlers[_event_cls] = lambda event: event.event_id


__all__ = ["handlers"]
