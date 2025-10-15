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
from cpython cimport PyErr_CheckSignals, PyGILState_Ensure, PyGILState_Release, PyObject


# include "./base.pxi"
# include "./type_info.pxi"
# include "./dtype.pxi"
# include "./device.pxi"
# include "./object.pxi"
# include "./error.pxi"
# include "./string.pxi"
# include "./tensor.pxi"
# include "./function.pxi"

cdef extern from "tvm/ffi/extra/c_env_api.h":
    ctypedef void* TVMFFIStreamHandle

    int TVMFFIEnvRegisterCAPI(const char* name, void* ptr) nogil
    void* TVMFFIEnvGetStream(int32_t device_type, int32_t device_id) nogil
    int TVMFFIEnvSetStream(int32_t device_type, int32_t device_id, TVMFFIStreamHandle stream,
                           TVMFFIStreamHandle* opt_out_original_stream) nogil


cdef _init_env_api():
    # Initialize env api for signal handling
    # Also registers the gil state release and ensure as PyErr_CheckSignals
    # function is called with gil released and we need to regrab the gil
    TVMFFIEnvRegisterCAPI(c_str("PyErr_CheckSignals"), <void*>PyErr_CheckSignals)

