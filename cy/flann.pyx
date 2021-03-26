import numpy as np
cimport numpy as np
np.import_array()

from cy.cflann cimport *
from enum import Enum
import os.path as osp
import os


class Algorithm(Enum):
    linear = FLANN_INDEX_LINEAR
    kdtree = FLANN_INDEX_KDTREE
    kmeans = FLANN_INDEX_KMEANS
    composite = FLANN_INDEX_COMPOSITE
    kdtree_single = FLANN_INDEX_KDTREE_SINGLE
    hierarchical = FLANN_INDEX_HIERARCHICAL
    lsh = FLANN_INDEX_LSH
    # kdtree_cuda = FLANN_INDEX_KDTREE_CUDA
    saved = FLANN_INDEX_SAVED
    autotuned = FLANN_INDEX_AUTOTUNED

class CentersInit(Enum):
    random = FLANN_CENTERS_RANDOM
    gonzales = FLANN_CENTERS_GONZALES
    kmeanspp = FLANN_CENTERS_KMEANSPP
    groupwise = FLANN_CENTERS_GROUPWISE


class LogLevel(Enum):
    none = FLANN_LOG_NONE
    fatal = FLANN_LOG_FATAL
    error = FLANN_LOG_ERROR
    warn = FLANN_LOG_WARN
    info = FLANN_LOG_INFO
    debug = FLANN_LOG_DEBUG

class Distance(Enum):
    euclidean = FLANN_DIST_EUCLIDEAN
    l2 = FLANN_DIST_L2
    manhattan = FLANN_DIST_MANHATTAN
    l1 = FLANN_DIST_L1
    minkowski = FLANN_DIST_MINKOWSKI
    max = FLANN_DIST_MAX
    hist_intersect = FLANN_DIST_HIST_INTERSECT
    hellinger = FLANN_DIST_HELLINGER
    chi_square = FLANN_DIST_CHI_SQUARE
    kullback_leibler = FLANN_DIST_KULLBACK_LEIBLER
    hamming = FLANN_DIST_HAMMING
    hamming_lut = FLANN_DIST_HAMMING_LUT
    hamming_popcnt = FLANN_DIST_HAMMING_POPCNT
    l2_simple = FLANN_DIST_L2_SIMPLE

class Datatype(Enum):
    none = FLANN_NONE
    int8 = FLANN_INT8
    int16 = FLANN_INT16
    int32 = FLANN_INT32
    int64 = FLANN_INT64
    uint8 = FLANN_UINT8
    uint16 = FLANN_UINT16
    uint32 = FLANN_UINT32
    uint64 = FLANN_UINT64
    float32 = FLANN_FLOAT32
    float64 = FLANN_FLOAT64


class Checks(Enum):
    unlimited = FLANN_CHECKS_UNLIMITED
    autotuned = FLANN_CHECKS_AUTOTUNED

cdef class FlannParameters:

    cdef FLANNParameters c_params

    def __cinit__(self, **params):
        self.c_params = DEFAULT_FLANN_PARAMETERS

    def __init__(self, **params):
        for key, value in params.items():
            self.__setattr__(key, value)

    cpdef FLANNParameters get_params(self):
        return self.c_params

    cdef FLANNParameters * ptr(self):
        return & self.c_params

    @property
    def algorithm(self):
        return Algorithm(self.c_params.algorithm)

    @algorithm.setter
    def algorithm(self, item):
        self.c_params.algorithm = Algorithm(item).value

    # search time parameters

    @property
    def checks(self):
        return self.c_params.checks

    @checks.setter
    def checks(self, item):
        self.c_params.checks = int(item)

    @property
    def eps(self):
        return self.c_params.eps

    @eps.setter
    def eps(self, value):
        self.c_params.eps = float(value)

    @property
    def sorted(self):
        return False if self.c_params.sorted == 0 else True

    @sorted.setter
    def sorted(self, value):
        self.c_params.sorted = int(bool(value))

    @property
    def max_neighbors(self):
        return self.c_params.max_neighbors

    @max_neighbors.setter
    def max_neighbors(self, value):
        self.c_params.max_neighbors = int(value)

    @property
    def cores(self):
        return self.c_params.cores

    @cores.setter
    def cores(self, value):
        self.c_params.cores = int(value)

    # kdtree index parameters

    @property
    def trees(self):
        return self.c_params.trees

    @trees.setter
    def trees(self, value):
        self.c_params.trees = int(value)

    @property
    def leaf_max_size(self):
        return self.c_params.leaf_max_size

    @leaf_max_size.setter
    def leaf_max_size(self, value):
        self.c_params.leaf_max_size = int(value)

    # kmeans index parameters

    @property
    def branching(self):
        return self.c_params.branching

    @branching.setter
    def branching(self, value):
        self.c_params.branching = int(value)

    @property
    def iterations(self):
        return self.c_params.iterations

    @iterations.setter
    def iterations(self, value):
        self.c_params.iterations = int(value)

    @property
    def centers_init(self):
        return CentersInit(self.c_params.centers_init)

    @centers_init.setter
    def centers_init(self, value):
        self.c_params.centers_init = CentersInit(value).value

    @property
    def cb_index(self):
        return self.c_params.cb_index

    @cb_index.setter
    def cb_index(self, value):
        self.c_params.cb_index = float(value)

    # autotuned index parameters

    @property
    def target_precision(self):
        return self.c_params.target_precision

    @target_precision.setter
    def target_precision(self, value):
        self.c_params.target_precision = float(value)

    @property
    def build_weight(self):
        return self.c_params.build_weight

    @build_weight.setter
    def build_weight(self, value):
        self.c_params.build_weight = float(value)

    @property
    def memory_weight(self):
        return self.c_params.memory_weight

    @memory_weight.setter
    def memory_weight(self, value):
        self.c_params.memory_weight = float(value)

    @property
    def sample_fraction(self):
        return self.c_params.sample_fraction

    @sample_fraction.setter
    def sample_fraction(self, value):
        self.c_params.sample_fraction = float(value)

    @property
    def log_level(self):
        return LogLevel(self.c_params.log_level)

    @log_level.setter
    def log_level(self, value):
        self.c_params.log_level = LogLevel(value).value

    @property
    def random_seed(self):
        return self.c_params.random_seed

    @random_seed.setter
    def random_seed(self, value):
        self.c_params.random_seed = int(value)



cdef float * _to_float_ptr(ds):
    cpdef float [::1] float_view = ds.ravel()
    return &float_view[0]

cdef double * _to_double_ptr(ds):
    cpdef double [::1] double_view = ds.ravel()
    return &double_view[0]

cdef int * _to_int_ptr(ds):
    cpdef int [::1] int_view = ds.ravel()
    return &int_view[0]

cdef unsigned char * _to_byte_ptr(ds):
    cpdef unsigned char [::1] byte_view = ds.ravel()
    return &byte_view[0]


def standardize_array(arr, dtype, ndim=2):
    """Ensure input array in 2-dimensional, contiguous and alligned. Return the array"""

    supported_types = [np.uint8, np.int32, np.float32, np.float64]
    # The type must be one of those supported by the FLANN library
    assert dtype in supported_types, f"Given type {dtype} is not supported " \
                                           f"(must be one of {[x.__name__ for x in supported_types]})"

    arr = np.require(arr, requirements=['C_CONTIGUOUS', 'ALIGNED'], dtype=dtype)
    if arr.ndim != ndim:
            raise ValueError(f"Provided dataset must be {ndim} dimensional. Received shape: {arr.shape}")

    return arr

cdef class FlannIndex:

    cdef flann_index_t _index_ptr
    cdef bint          _index_built
    cdef _log_level

    cdef _dtype
    cdef _dataset
    cdef _labels

    def __cinit__(self, dataset, dtype=np.float32, log_level=None, labels=None, _build_index=True, **kwargs):

        # Store the dataset itself
        self._dataset = standardize_array(dataset, dtype)

        self._dtype = dtype
        self._index_built = False

        # Set log level (if given otherwise default)
        if log_level is not None:
            log_level = LogLevel(log_level)
        else:
            log_level = LogLevel.warn

        flann_log_verbosity(LogLevel(log_level).value)
        self._log_level = LogLevel(log_level).value

        # Store the labels
        if labels is None:
            self._labels = np.full(self._dataset.shape[0], None)

        else:
            assert len(labels) == self._dataset.shape[0], "Num labels must match the number of data points"
            self._labels = np.asarray(labels)

        # Create the index
        cdef int rows = self._dataset.shape[0]
        cdef int cols = self._dataset.shape[1]

        cdef FLANNParameters * c_params = FlannParameters(**kwargs).ptr()
        cdef float speedup

        if _build_index:

            if self._dtype == np.float32:
                self._index_ptr = flann_build_index_float(_to_float_ptr(self._dataset), rows, cols, &speedup, c_params)

            elif self._dtype == np.float64:
                self._index_ptr = flann_build_index_double(_to_double_ptr(self._dataset), rows, cols, &speedup, c_params)

            elif self._dtype == np.int32:
                self._index_ptr = flann_build_index_int(_to_int_ptr(self._dataset), rows, cols, &speedup, c_params)

            elif self._dtype == np.uint8:
                self._index_ptr = flann_build_index_byte(_to_byte_ptr(self._dataset), rows, cols, &speedup, c_params)

            else:
                raise TypeError(f"Unsupported type {self._dtype}")

            self._dataset.flags.writeable = False


    @property
    def log_level(self):
        return LogLevel(self._log_level)

    @log_level.setter
    def log_level(self, log_level):
        self._log_level = LogLevel(log_level).value
        flann_log_verbosity(self._log_level)

    @property
    def num_features(self):

        if self._dtype == np.float32:
            return flann_veclen_float(self._index_ptr)

        elif self._dtype == np.float64:
            return flann_veclen_double(self._index_ptr)

        elif self._dtype == np.int32:
            return flann_veclen_int(self._index_ptr)

        elif self._dtype == np.uint8:
            return flann_veclen_byte(self._index_ptr)

        else:
            raise TypeError(f"Unsupported type {self._dtype}")

    def __len__(self):

        if self._dtype == np.float32:
            return flann_size_float(self._index_ptr)

        elif self._dtype == np.float64:
            return flann_size_double(self._index_ptr)

        elif self._dtype == np.int32:
            return flann_size_int(self._index_ptr)

        elif self._dtype == np.uint8:
            return flann_size_byte(self._index_ptr)

        else:
            raise TypeError(f"Unsupported type {self._dtype}")

    @property
    def used_memory(self):

        if not self._index_built:
            return 0

        if self._dtype == np.float32:
            return flann_used_memory_float(self._index_ptr)

        elif self._dtype == np.float64:
            return flann_used_memory_double(self._index_ptr)

        elif self._dtype == np.int32:
            return flann_used_memory_int(self._index_ptr)

        elif self._dtype == np.uint8:
            return flann_used_memory_byte(self._index_ptr)

        else:
            raise TypeError(f"Unsupported type {self._dtype}")


    def save(self, filename):

        from visual.src.data.utils.json.IO import json_dump

        # Make sure the output has the right extension
        if not filename.endswith('.flann'):
            filename = filename + '.flann'

        # Create to save the results
        os.makedirs(osp.dirname(filename), exist_ok=True)

        cdef bytes py_bytes = filename.encode()
        cdef char* filename_ptr = py_bytes
        cdef int status

        if self._dtype == np.float32:
            status = flann_save_index_float(self._index_ptr, filename_ptr)

        elif self._dtype == np.float64:
            status = flann_save_index_double(self._index_ptr, filename_ptr)

        elif self._dtype == np.int32:
            status = flann_save_index_int(self._index_ptr, filename_ptr)

        elif self._dtype == np.uint8:
            status = flann_save_index_byte(self._index_ptr, filename_ptr)

        else:
            raise TypeError(f"Unsupported type {self._dtype}")

        if status != 0:
            raise IOError(f"Index Save failed with status {status}")

        dataset_filename = filename + '.npy'
        with open(dataset_filename, 'wb') as fp:
            np.save(fp, self._dataset)

        labels_filename = filename + '.json'
        with open(labels_filename, 'w') as fp:
            json_dump(self._labels.tolist(), fp)


    cpdef _load_index(self, filename):

        cdef bytes py_bytes = filename.encode()
        cdef char* filename_ptr = py_bytes
        cdef int status

        cdef int rows = self._dataset.shape[0]
        cdef int cols = self._dataset.shape[1]

        cdef flann_index_t index_ptr

        if self._dtype == np.float32:
            self._index_ptr = flann_load_index_float(filename_ptr, _to_float_ptr(self._dataset), rows, cols)

        elif self._dtype == np.float64:
            self._index_ptr = flann_load_index_double(filename_ptr, _to_double_ptr(self._dataset), rows, cols)

        elif self._dtype == np.int32:
            self._index_ptr = flann_load_index_int(filename_ptr, _to_int_ptr(self._dataset), rows, cols)

        elif self._dtype == np.uint8:
            self._index_ptr = flann_load_index_byte(filename_ptr, _to_byte_ptr(self._dataset), rows, cols)

        else:
            raise TypeError(f"Unsupported type {self._dtype}")

        self._dataset.flags.writeable = False

    @classmethod
    def load(cls, filename):

        from visual.src.data.utils.json.IO import json_load

        # Make sure the output has the right extension
        if not filename.endswith('.flann'):
            filename = filename + '.flann'

        if not osp.isfile(filename):
            raise IOError(f"No such file {filename}")

        dataset_filename = filename + '.npy'
        with open(dataset_filename, 'rb') as fp:
            dataset = np.load(fp)

        labels_filename = filename + '.json'
        with open(labels_filename, 'r') as fp:
            labels = json_load(fp)

        dtype = dataset.dtype
        dataset = standardize_array(dataset, dtype=dtype)

        flann_index = cls(dataset=dataset, dtype=dtype, labels=labels, _build_index=False)
        flann_index._load_index(filename)

        return flann_index

    def __dealloc__(self):

        cdef FLANNParameters c_params
        c_params.log_level = int(LogLevel(self._log_level).value)

        cdef int status

        if self._dtype == np.float32:
            status = flann_free_index_float(self._index_ptr, &c_params)

        elif self._dtype == np.float64:
            status = flann_free_index_double(self._index_ptr, &c_params)

        elif self._dtype == np.int32:
            status = flann_free_index_int(self._index_ptr, &c_params)

        else:
            status = flann_free_index_byte(self._index_ptr, &c_params)

    def __getitem__(self, idx):
        return self._dataset[idx]

    def get_data_point(self, item):

        assert isinstance(item, int), "Only integer direct indexing is supported"

        if item < 0 or item >= len(self):
            raise IndexError("Index out of range")

        cdef unsigned int point_id = int(item)

        cdef float * float_ptr
        cdef double * double_ptr
        cdef unsigned char * byte_ptr
        cdef int * int_ptr
        cdef int num_features = self.num_features


        if self._dtype == np.float32:
            float_ptr = flann_get_point_float(self._index_ptr, point_id)
            result = np.asarray(<float[:self.num_features]> float_ptr, dtype=self._dtype, order='c')

        elif self._dtype == np.float64:
            double_ptr = flann_get_point_double(self._index_ptr, point_id)
            result = np.asarray(<double[:self.num_features]> double_ptr, dtype=self._dtype, order='c')

        elif self._dtype == np.uint8:
            byte_ptr = flann_get_point_byte(self._index_ptr, point_id)
            result = np.asarray(<unsigned char [:self.num_features]> byte_ptr, dtype=self._dtype, order='c')

        elif self._dtype == np.int32:
            int_ptr = flann_get_point_int(self._index_ptr, point_id)
            result = np.asarray(<int[:self.num_features]> int_ptr, dtype=self._dtype, order='c')

        else:
            raise TypeError(f"Unsupported type {self._dtype}")

        # Make the data readonly
        result.flags.writeable = False
        return result

    def nn_search(self, testset, k=1, **kwargs):

        testset = standardize_array(testset, self._dtype)

        if testset.shape[1] != self.num_features:
            raise ValueError(f"Test vectors must be of shape (?, {self.num_features})")

        cdef int trows = testset.shape[0]
        cdef int nn = int(k)

        indices = standardize_array(np.zeros(shape=(trows, nn), dtype=np.int32, order='c'), dtype=np.int32)

        if self._dtype == np.float64:
            dists = standardize_array(np.zeros(shape=(trows, nn), dtype=np.float64, order='c'), dtype=np.float64)
        else:
            dists = standardize_array(np.zeros(shape=(trows, nn), dtype=np.float32, order='c'), dtype=np.float32)

        cdef FLANNParameters * c_params = FlannParameters(**kwargs).ptr()
        cdef int status

        if self._dtype == np.float32:
            status = flann_find_nearest_neighbors_index_float(self._index_ptr,
                                                        _to_float_ptr(testset),
                                                        trows,
                                                        _to_int_ptr(indices),
                                                        _to_float_ptr(dists),
                                                        nn, c_params)
        elif self._dtype == np.float64:
            status = flann_find_nearest_neighbors_index_double(self._index_ptr,
                                                        _to_double_ptr(testset),
                                                        trows,
                                                        _to_int_ptr(indices),
                                                        _to_double_ptr(dists),
                                                        nn, c_params)

        elif self._dtype == np.int32:
            status = flann_find_nearest_neighbors_index_int(self._index_ptr,
                                                        _to_int_ptr(testset),
                                                        trows,
                                                        _to_int_ptr(indices),
                                                        _to_float_ptr(dists),
                                                        nn, c_params)

        elif self._dtype == np.uint8:
            status = flann_find_nearest_neighbors_index_byte(self._index_ptr,
                                                        _to_byte_ptr(testset),
                                                        trows,
                                                        _to_int_ptr(indices),
                                                        _to_float_ptr(dists),
                                                        nn, c_params)

        else:
            raise TypeError(f"Unsupported type {self._dtype}")

        if status != 0:
            raise RuntimeError(f"Nearest neighbor search failed with status {status}")

        return indices, dists

    def radius_search(self, point, radius, max_nn=10, **kwargs):

        point = standardize_array(point, self._dtype, ndim=1)

        if point.shape[0] != self.num_features:
            raise ValueError(f"Test point must be of shape (?, {self.num_features})")

        indices = standardize_array(np.zeros(shape=max_nn, dtype=np.int32, order='c'), dtype=np.int32, ndim=1)

        if self._dtype == np.float64:
            dists = standardize_array(np.zeros(shape=max_nn, dtype=np.float64, order='c'), dtype=np.float64, ndim=1)
        else:
            dists = standardize_array(np.zeros(shape=max_nn, dtype=np.float32, order='c'), dtype=np.float32, ndim=1)

        cdef FLANNParameters * c_params = FlannParameters(**kwargs).ptr()
        cdef int num_nn

        if self._dtype == np.float32:
            num_nn = flann_radius_search_float(self._index_ptr,
                                               _to_float_ptr(point),
                                               _to_int_ptr(indices),
                                               _to_float_ptr(dists),
                                               int(max_nn),
                                               float(radius),
                                               c_params)

        elif self._dtype == np.float64:
            num_nn = flann_radius_search_double(self._index_ptr,
                                                _to_double_ptr(point),
                                                _to_int_ptr(indices),
                                                _to_double_ptr(dists),
                                                int(max_nn),
                                                float(radius),
                                                c_params)

        elif self._dtype == np.int32:
            num_nn = flann_radius_search_int(self._index_ptr,
                                             _to_int_ptr(point),
                                             _to_int_ptr(indices),
                                             _to_float_ptr(dists),
                                             int(max_nn),
                                             float(radius),
                                             c_params)

        elif self._dtype == np.uint8:
            num_nn = flann_radius_search_byte(self._index_ptr,
                                              _to_byte_ptr(point),
                                              _to_int_ptr(indices),
                                              _to_float_ptr(dists),
                                              int(max_nn),
                                              float(radius),
                                              c_params)

        else:
            raise TypeError(f"Unsupported type {self._dtype}")

        if num_nn < 0:
            raise RuntimeError(f"Radius search failed with status {num_nn}")


        dists = dists[:num_nn]
        indices = indices[:num_nn]

        return indices, dists


    @property
    def dtype(self):
        return self._dtype

    @property
    def dataset(self):
        return self._dataset

    @property
    def labels(self):
        return self._labels
