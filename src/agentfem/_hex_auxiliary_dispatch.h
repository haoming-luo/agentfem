/* SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
 * SPDX-License-Identifier: Apache-2.0
 * Disjoint-cell dispatch. Optional instance executor, no retained scratch or
 * material/state owner. Every batch completes before publication or release.
 */
typedef struct {
    int operation, n, invalid;
    Py_ssize_t nodes;
    const double *in[10];
    const int32_t *indices;
    double *out[5];
    int *status;
} auxiliary_job;

static void *execute_auxiliary(void *raw) {
    auxiliary_job *j = raw;
    if (j->operation == 0) {
        hex_gradient(j->n, j->in[0], j->in[1], j->in[2], j->in[3],
                     (const int *)j->indices, j->out[0], j->status);
        for (int k = 0; k < j->n; k++) if (!j->status[k])
            j->status[k] = hex_affine_admitted(j->in[0]+24*k, j->in[1]+24*k,
                j->out[0]+9*k, j->in[3][k], (const int *)j->indices);
    } else if (j->operation == 1) {
        hex_bulk_response(j->n, j->in[0], j->in[1], j->in[2], j->in[3],
            j->in[4], j->in[5], j->in[6], j->in[7], j->in[8], j->in[9],
            j->out[0], j->out[1], j->out[2], j->out[3], j->out[4], j->status);
        for (int k = 0; k < j->n; k++) if (j->status[k]) {
            for (int q = 0; q < 24; q++) j->out[0][24*k+q] = NAN;
            for (int q = 1; q < 5; q++) j->out[q][k] = NAN;
        }
    } else {
        for (Py_ssize_t k = 0; k < (Py_ssize_t)j->n * 8; k++) {
            int32_t node = j->indices[k];
            if (node < 0 || node >= j->nodes) { j->invalid = 1; continue; }
            memcpy(j->out[0] + 3*k, j->in[0] + 3*(Py_ssize_t)node, 3*sizeof(double));
        }
    }
    return NULL;
}

static int dispatch_auxiliary(auxiliary_job *jobs, int workers, NativePool *pool) {
    if (workers == 1) { execute_auxiliary(jobs); return 0; }
    if(pool) { native_pool_dispatch(pool,jobs,sizeof(auxiliary_job),workers,execute_auxiliary);return 0; }
    pthread_t threads[16]; int started = 0, failed = 0;
    for (int i = 0; i < workers; i++) {
        if (pthread_create(&threads[started], NULL, execute_auxiliary, &jobs[i])) {
            failed = 1; break;
        }
        started++;
    }
    for (int i = 0; i < started; i++) pthread_join(threads[i], NULL);
    return failed;
}

static PyObject *auxiliary(PyObject *self, PyObject *args) {
    int operation, workers; PyObject *inputs, *result = NULL, *outputs[6] = {NULL};
    Py_buffer buffers[10] = {{0}};
    PyObject *executor=Py_None;NativePool *pool=NULL;int acquired=0;
    if (!PyArg_ParseTuple(args, "iOi|O", &operation, &inputs, &workers,&executor)) return NULL;
    int count = operation == 0 ? 5 : operation == 1 ? 10 : operation == 2 ? 2 : 0;
    if (!count || workers < 1 || workers > 16 || !PyTuple_Check(inputs)
        || PyTuple_GET_SIZE(inputs) != count) {
        PyErr_SetString(PyExc_ValueError, "Invalid native auxiliary configuration"); return NULL;
    }
    if(executor!=Py_None) {
        if(Py_TYPE(executor)!=&NativePoolType) {
            PyErr_SetString(PyExc_TypeError,"Expected an instance native executor");return NULL;
        }
        pool=(NativePool *)executor;
        if(native_pool_acquire(pool,workers))return NULL;
        acquired=1;
    }
    for (int i = 0; i < count; i++) {
        int integer = (operation == 0 && i == 4) || (operation == 2 && i == 1);
        if (PyObject_GetBuffer(PyTuple_GET_ITEM(inputs, i), &buffers[i],
                              PyBUF_C_CONTIGUOUS | PyBUF_FORMAT)) goto cleanup;
        if (!buffers[i].format || strcmp(buffers[i].format, integer ? "i" : "d")
            || buffers[i].itemsize != (integer ? 4 : 8)
            || (uintptr_t)buffers[i].buf % (integer ? 4 : 8)) {
            PyErr_SetString(PyExc_ValueError, "Aligned float64/int32 auxiliary buffers required"); goto cleanup;
        }
    }
    Py_ssize_t n = buffers[0].len / (operation == 0 ? 192 : 72);
    if (operation == 2) n = buffers[1].len / 32;
    if (n < 0 || n > INT_MAX / 24 || n > PY_SSIZE_T_MAX / 192) {
        PyErr_SetString(PyExc_ValueError, "Invalid auxiliary cell count"); goto cleanup;
    }
    const int gradient_width[4] = {24, 24, 24, 1};
    const int bulk_width[10] = {9, 24, 24, 8, 1, 1, 1, 1, 1, 1};
    const int *width = operation == 0 ? gradient_width : bulk_width;
    if (operation != 2) {
        for (int i = 0; i < (operation == 0 ? 4 : count); i++) {
            if (buffers[i].len != n * width[i] * 8) {
                PyErr_SetString(PyExc_ValueError, "Incompatible auxiliary buffer lengths"); goto cleanup;
            }
        }
        if (operation == 0) {
            if (buffers[4].len != 24 * 4) {
                PyErr_SetString(PyExc_ValueError, "Expected 12 Hex8 coordinate edges"); goto cleanup;
            }
            const int32_t *edges = buffers[4].buf;
            for (int i = 0; i < 24; i++) if (edges[i] < 0 || edges[i] > 7) {
                PyErr_SetString(PyExc_ValueError, "Invalid Hex8 edge node"); goto cleanup;
            }
        }
    } else if (buffers[0].ndim != 2 || buffers[0].shape[1] != 3
               || buffers[1].ndim != 2 || buffers[1].shape[1] != 8
               || buffers[1].len != n * 32) {
        PyErr_SetString(PyExc_ValueError, "Gather requires nodal vectors and Hex8 int32 connectivity"); goto cleanup;
    }
    int output_count = operation == 0 ? 2 : operation == 1 ? 6 : 1;
    for (int i = 0; i < output_count; i++) {
        Py_ssize_t bytes = (operation != 2 && i == output_count - 1) ? n*sizeof(int)
            : n * (i == 0 ? (operation == 0 ? 9 : 24) : 1) * 8;
        outputs[i] = PyBytes_FromStringAndSize(NULL, bytes);
        if (!outputs[i]) goto cleanup;
    }
    /* Small batches execute in the calling thread. No nested BLAS/OpenMP use. */
    if (n < 8192) workers = 1;
    auxiliary_job jobs[16] = {{0}};
    for (int i = 0; i < workers; i++) {
        Py_ssize_t start = n*i/workers, end = n*(i+1)/workers;
        auxiliary_job *j = &jobs[i]; j->operation = operation; j->n = (int)(end-start);
        if (operation == 2) {
            j->in[0] = buffers[0].buf; j->nodes = buffers[0].shape[0];
            j->indices = (const int32_t *)buffers[1].buf + start*8;
        } else {
            for (int k = 0; k < (operation == 0 ? 4 : count); k++)
                j->in[k] = (const double *)buffers[k].buf + start*width[k];
            if (operation == 0) j->indices = buffers[4].buf;
            j->status = (int *)PyBytes_AS_STRING(outputs[output_count-1]) + start;
        }
        for (int k = 0; k < (operation == 2 ? 1 : output_count-1); k++)
            j->out[k] = (double *)PyBytes_AS_STRING(outputs[k])
                + start * (k == 0 ? (operation == 0 ? 9 : 24) : 1);
    }
    int failed;
    Py_BEGIN_ALLOW_THREADS
    failed = dispatch_auxiliary(jobs, workers,pool);
    Py_END_ALLOW_THREADS
    if (PyErr_CheckSignals()) goto cleanup;
    if (failed) { PyErr_SetString(PyExc_RuntimeError, "Native auxiliary thread creation failed"); goto cleanup; }
    for (int i = 0; i < workers; i++) if (jobs[i].invalid) {
        PyErr_SetString(PyExc_ValueError, "Gather connectivity is outside nodal storage"); goto cleanup;
    }
    result = PyTuple_New(output_count);
    if (result) for (int i = 0; i < output_count; i++) PyTuple_SET_ITEM(result, i, Py_NewRef(outputs[i]));
cleanup:
    if(acquired)pool->busy=0;
    for (int i = 0; i < 6; i++) Py_XDECREF(outputs[i]);
    for (int i = 0; i < 10; i++) if (buffers[i].obj) PyBuffer_Release(&buffers[i]);
    return result;
}
