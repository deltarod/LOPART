#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_9_API_VERSION
#include <numpy/arrayobject.h>
#include "LOPART.h"


static PyObject * LOPARTInterface(PyObject *self, PyObject *args)
{
     PyArrayObject *inputData, *input_label_start, *input_label_end, *input_label_changes;

    int lenData, lenLabels, numUpdates;
    double penalty_unlabeled, penalty_labeled;

    if(!PyArg_ParseTuple(args, "O!iO!O!O!iddi",
                         &PyArray_Type, &inputData,
                         &lenData,
                         &PyArray_Type, &input_label_start,
                         &PyArray_Type, &input_label_end,
                         &PyArray_Type, &input_label_changes,
                         &lenLabels,
                         &penalty_unlabeled,
                         &penalty_labeled,
                         &numUpdates))
    {
        PyErr_SetString(PyExc_TypeError, "Parse Tuple Error");
        return NULL;
    }
    if(PyArray_TYPE(inputData) != NPY_DOUBLE)
    {
        PyErr_SetString(PyExc_TypeError, "Input Data must be numpy.ndarray type double");
        return NULL;
    }
    if(PyArray_TYPE(input_label_start) != NPY_INT)
    {
        PyErr_SetString(PyExc_TypeError, "Input Label Start Data must be numpy.ndarray type int");
        return NULL;
    }
    if(PyArray_TYPE(input_label_end) != NPY_INT)
    {
        PyErr_SetString(PyExc_TypeError, "Input Label End Data must be numpy.ndarray type int");
        return NULL;
    }
    if(PyArray_TYPE(input_label_changes) != NPY_INT)
    {
        PyErr_SetString(PyExc_TypeError, "Input Label changes Data must be numpy.ndarray type int");
        return NULL;
    }

    // Input Data Formatting
    double *inputDataA = (double*)PyArray_DATA(inputData);
    int *input_label_startA = (int*)PyArray_DATA(input_label_start);
    int *input_label_endA = (int*)PyArray_DATA(input_label_end);
    int *input_label_changesA = (int*)PyArray_DATA(input_label_changes);

    // Output Data Formatting
    npy_intp col_dim = PyArray_DIM(inputData, 0);

    PyArrayObject *out_cumsum, *out_change_candidates, *out_cost_candidates, *out_cost, *out_mean, *out_last_change;

    out_cumsum = PyArray_ZEROS(1, &col_dim, NPY_DOUBLE, 0);
    double *out_cumsumA = (double*)PyArray_DATA(out_cumsum);

    out_change_candidates = PyArray_ZEROS(1, &col_dim, NPY_INT, 0);
    int *out_change_candidatesA = (int*)PyArray_DATA(out_change_candidates);

    out_cost_candidates = PyArray_ZEROS(1, &col_dim, NPY_DOUBLE, 0);
    double *out_cost_candidatesA = (double*)PyArray_DATA(out_cost_candidates);

    out_cost = PyArray_ZEROS(1, &col_dim, NPY_DOUBLE, 0);
    double *out_costA = (double*)PyArray_DATA(out_cost);

    out_mean = PyArray_ZEROS(1, &col_dim, NPY_DOUBLE, 0);
    double *out_meanA = (double*)PyArray_DATA(out_mean);

    out_last_change = PyArray_ZEROS(1, &col_dim, NPY_INT, 0);
    int *out_last_changeA = (int*)PyArray_DATA(out_last_change);

    int status = LOPART( inputDataA,
                         lenData,
                         input_label_startA,
                         input_label_endA,
                         input_label_changesA,
                         lenLabels,
                         penalty_unlabeled,
                         penalty_labeled,
                         numUpdates,
                         //inputs above, outputs below.
                         out_cumsumA,
                         out_change_candidatesA,
                         out_cost_candidatesA,
                         out_costA,
                         out_meanA,
                         out_last_changeA);
    if(status == ERROR_EACH_LABEL_START_MUST_BE_LESS_THAN_ITS_END)
    {
        PyErr_SetString(PyExc_ValueError,
                        "Start needs to be less than end");
    }
    if(status == ERROR_LABELED_NUMBER_OF_CHANGES_MUST_BE_0_OR_1)
    {
        PyErr_SetString(PyExc_ValueError,
                        "Labeled Changes need to be 1 or 0");
    }
    if(status == ERROR_EACH_LABEL_START_MUST_BE_ON_OR_AFTER_PREVIOUS_END)
    {
        PyErr_SetString(PyExc_ValueError,
                        "Label Start must be on or after previous end");
    }
    if(status == ERROR_LABEL_START_MUST_BE_ZERO_OR_LARGER)
    {
        PyErr_SetString(PyExc_ValueError,
                        "Label start must be zero or larger");
    }
    if(status == ERROR_LABEL_END_MUST_BE_LESS_THAN_N_DATA)
    {
        PyErr_SetString(PyExc_ValueError,
                        "label end must be less than N data");
    }
    if(status == ERROR_PENALTY_MUST_BE_NON_NEGATIVE)
    {
        PyErr_SetString(PyExc_ValueError,
                        "Penalty can't be negative");
    }
    if(status == ERROR_NO_DATA)
    {
        PyErr_SetString(PyExc_ValueError,
                        "No Data");
    }
    if(status == ERROR_DATA_MUST_BE_FINITE)
    {
        PyErr_SetString(PyExc_ValueError,
                        "Data must be finite");
    }
    if(status != 0){
        return NULL;
    }

    return Py_BuildValue("{s:O,s:O,s:O,s:O}",
             "cost_candidates", out_change_candidates,
             "cost_optimal", out_cost_candidates,
             "mean", out_mean,
             "last_change", out_last_change);
}

static PyMethodDef Methods[] = {
        {"interface", LOPARTInterface, METH_VARARGS,
                        "Runs Labeled Optimal Partitioning"},
        {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduleDef =
        {
        PyModuleDef_HEAD_INIT,
        "LOPARTInterface",
        "A Python extension for LOPART",
        -1,
        Methods
        };


PyMODINIT_FUNC
PyInit_LOPARTInterface(void)
{
    PyObject *module;
    module = PyModule_Create(&moduleDef);
    if(module == NULL) return NULL;
    import_array();//necessary from numpy otherwise we crash with segfault
    return module;
}



