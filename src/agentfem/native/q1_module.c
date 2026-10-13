/* SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
 * SPDX-License-Identifier: Apache-2.0 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <float.h>
#if DBL_MANT_DIG != 53 || DBL_MAX_EXP != 1024
#error Q1 predicates require IEEE binary64.
#endif
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__)
#error Q1 predicates require strict arithmetic.
#endif
/* Internal ctypes symbols are called only from validated geometry owners.
 * This module exposes no raw-pointer functions in the public Python API. */
static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "_q1_native",
    "Stateless bounded Q1 geometry; GMP rational recovery.", -1, NULL};
PyMODINIT_FUNC PyInit__q1_native(void) {
    PyObject *result=PyModule_Create(&module);
    if(result && PyModule_AddStringConstant(result,"projection_policy",
        "scale-conditioned-margin+bernstein-gmp-v1")<0) {
        Py_DECREF(result);return NULL;
    }
    return result;
}
