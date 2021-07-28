# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# License); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
# Author: tuanhe
#

# 0. clear var
UNSET (_DEV_AIPU_HEADER_PATH)
UNSET (_AIPU_BASE_SOURCE)
UNSET (_AIPU_OPS_SOURCE)
UNSET (_DEV_AIPU_DEVICE_SOURCE)
UNSET (_DEV_AIPU_COMPILER_DEFINES)
UNSET (_DEV_AIPU_COMPILER_OPTIONS)
UNSET (_DEV_AIPU_LINKER_OPTIONS)
UNSET (_DEV_AIPU_LINK_LIBRARIES)

# 1.  set source root path
SET(_AIPU_ROOT ${CMAKE_SOURCE_DIR}/source/device/aipu)
set(AIPUSDK_ROOT ${CMAKE_SOURCE_DIR}/../sdk)
set(AIPUSDK_INCLUDE_DIRS ${AIPUSDK_ROOT}/include)
set(AIPUSDK_LIB_DIRS ${AIPUSDK_ROOT}/lib)


# 2.  add header file path
LIST (APPEND _DEV_AIPU_HEADER_PATH      ${_AIPU_ROOT})
LIST (APPEND _DEV_AIPU_HEADER_PATH      ${AIPUSDK_INCLUDE_DIRS})


# 3.  add linking lib searching path
LIST (APPEND _DEV_AIPU_LINK_PATH        ${AIPUSDK_LIB_DIRS})


# 4.  add source files
AUX_SOURCE_DIRECTORY("${_AIPU_ROOT}"    _AIPU_BASE_SOURCE)
AUX_SOURCE_DIRECTORY("${_AIPU_ROOT}/op" _AIPU_OPS_SOURCE)
LIST (APPEND _DEV_AIPU_DEVICE_SOURCE    ${_AIPU_BASE_SOURCE})
LIST (APPEND _DEV_AIPU_DEVICE_SOURCE    ${_AIPU_OPS_SOURCE})


# 5.  add build options for cpu device
# 5.1 is a gcc or clang like compiler
IF (TENGINE_COMPILER_GCC OR TENGINE_COMPILER_CLANG)
ENDIF()


# 5.2 is Microsoft Visual C++
IF (TENGINE_COMPILER_MSVC)
ENDIF()


# 6.  add link options


# 7.  add link libs
LIST (APPEND _DEV_AIPU_LINK_LIBRARIES   aipubuildtool)
LIST (APPEND _DEV_AIPU_LINK_LIBRARIES   aipu_xlib)


# 8. set all to cmake cache
SET (TENGINE_AIPU_HEADER_PATH       ${_DEV_AIPU_HEADER_PATH}        CACHE INTERNAL  "Tengine AIPU device header files searching path"   FORCE)
SET (TENGINE_AIPU_LINK_PATH         ${_DEV_AIPU_LINK_PATH}          CACHE INTERNAL  "Tengine AIPU device link libraries searching path" FORCE)
SET (TENGINE_AIPU_DEVICE_SOURCE     ${_DEV_AIPU_DEVICE_SOURCE}      CACHE INTERNAL  "Tengine AIPU device main source files"             FORCE)
SET (TENGINE_AIPU_COMPILER_DEFINES  ${_DEV_AIPU_COMPILER_DEFINES}   CACHE INTERNAL  "Tengine AIPU about compiler defines"               FORCE)
SET (TENGINE_AIPU_COMPILER_OPTIONS  ${_DEV_AIPU_COMPILER_OPTIONS}   CACHE INTERNAL  "Tengine AIPU about compiler options"               FORCE)
SET (TENGINE_AIPU_LINKER_OPTIONS    ${_DEV_AIPU_LINKER_OPTIONS}     CACHE INTERNAL  "Tengine AIPU about linker options"                 FORCE)
SET (TENGINE_AIPU_LINK_LIBRARIES    ${_DEV_AIPU_LINK_LIBRARIES}     CACHE INTERNAL  "Tengine AIPU about link libraries"                 FORCE)


# 9. install device option
INSTALL (FILES ${_AIPU_ROOT}/aipu_define.h DESTINATION include/tengine RENAME aipu_device.h)