# cython: freethreading_compatible = True
# cython: language_level=3
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
import ctypes
from libc.stdint cimport int32_t, int64_t, uint64_t, uint32_t, uint8_t, int16_t
from libc.string cimport memcpy
from libcpp.vector cimport vector
from cpython.bytes cimport PyBytes_AsStringAndSize, PyBytes_FromStringAndSize, PyBytes_AsString
from cpython cimport Py_INCREF, Py_DECREF, Py_REFCNT
from cpython cimport PyErr_CheckSignals, PyGILState_Ensure, PyGILState_Release, PyObject
from cpython cimport pycapsule, PyCapsule_Destructor
from cpython cimport PyErr_SetNone


# include "./base.pxi"
# include "./type_info.pxi"
# include "./dtype.pxi"
# include "./device.pxi"
# include "./object.pxi"
# include "./error.pxi"
# include "./string.pxi"
# include "./tensor.pxi"
# include "./function.pxi"

def dummy_function():
    return 0

#cdef extern from "tvm/ffi/c_api.h":
#    int TVMDLLDummyFunction()


cdef _init_env_api():
    # Initialize env api for signal handling
    # Also registers the gil state release and ensure as PyErr_CheckSignals
      # function is called with gil released and we need to regrab the gil
    #TVMDLLDummyFunction()


