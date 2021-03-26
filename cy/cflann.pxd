cdef extern from "flann/defines.h":

    # Nearest neighbour index algorithms
    cpdef enum flann_algorithm_t:
        FLANN_INDEX_LINEAR
        FLANN_INDEX_KDTREE
        FLANN_INDEX_KMEANS
        FLANN_INDEX_COMPOSITE
        FLANN_INDEX_KDTREE_SINGLE
        FLANN_INDEX_HIERARCHICAL
        FLANN_INDEX_LSH
        FLANN_INDEX_KDTREE_CUDA
        FLANN_INDEX_SAVED
        FLANN_INDEX_AUTOTUNED


    cpdef enum flann_centers_init_t:
        FLANN_CENTERS_RANDOM
        FLANN_CENTERS_GONZALES
        FLANN_CENTERS_KMEANSPP
        FLANN_CENTERS_GROUPWISE

    cpdef enum flann_log_level_t:
        FLANN_LOG_NONE
        FLANN_LOG_FATAL
        FLANN_LOG_ERROR
        FLANN_LOG_WARN
        FLANN_LOG_INFO
        FLANN_LOG_DEBUG

    cpdef enum flann_distance_t:
        FLANN_DIST_EUCLIDEAN
        FLANN_DIST_L2
        FLANN_DIST_MANHATTAN
        FLANN_DIST_L1
        FLANN_DIST_MINKOWSKI
        FLANN_DIST_MAX
        FLANN_DIST_HIST_INTERSECT
        FLANN_DIST_HELLINGER
        FLANN_DIST_CHI_SQUARE
        FLANN_DIST_KULLBACK_LEIBLER
        FLANN_DIST_HAMMING
        FLANN_DIST_HAMMING_LUT
        FLANN_DIST_HAMMING_POPCNT
        FLANN_DIST_L2_SIMPLE

    cpdef enum flann_datatype_t:
        FLANN_NONE
        FLANN_INT8
        FLANN_INT16
        FLANN_INT32
        FLANN_INT64
        FLANN_UINT8
        FLANN_UINT16
        FLANN_UINT32
        FLANN_UINT64
        FLANN_FLOAT32
        FLANN_FLOAT64


    cpdef enum flann_checks_t:
        FLANN_CHECKS_UNLIMITED
        FLANN_CHECKS_AUTOTUNED



cdef extern from "flann/flann.h":

    cdef struct FLANNParameters:

        flann_algorithm_t algorithm

        # search time parameters
        int checks;    # how many leafs (features) to check in one search
        float eps;     # eps parameter for eps-knn search
        int sorted;     # indicates if results returned by radius search should be sorted or not
        int max_neighbors;  # limits the maximum number of neighbors should be returned by radius search
        int cores;      # number of paralel cores to use for searching

        #  kdtree index parameters
        int trees;                 # number of randomized trees to use (for kdtree)
        int leaf_max_size;

        # kmeans index parameters
        int branching;             # branching factor (for kmeans tree)
        int iterations;            # max iterations to perform in one kmeans cluetering (kmeans tree)
        flann_centers_init_t centers_init  # algorithm used for picking the initial cluster centers for kmeans tree
        float cb_index;            # cluster boundary index. Used when searching the kmeans tree

        # autotuned index parameters
        float target_precision;    # precision desired (used for autotuning, -1 otherwise)
        float build_weight;        # build tree time weighting factor
        float memory_weight;       # index memory weigthing factor
        float sample_fraction;     # what fraction of the dataset to use for autotuning

        # LSH parameters
        unsigned int table_number_; # The number of hash tables to use
        unsigned int key_size_;     # The length of the key in the hash tables
        unsigned int multi_probe_level_; # Number of levels to use in multi-probe LSH, 0 for standard LSH

        # other parameters
        flann_log_level_t log_level;    # determines the verbosity of each flann function
        long random_seed;            # random seed to use

    ctypedef void* flann_index_t;

    extern FLANNParameters DEFAULT_FLANN_PARAMETERS;

    # Sets the log level used for all flann functions (unless
    # specified in FLANNParameters for each call
    #
    # Params:
    # level = verbosity level
    void flann_log_verbosity(int level);


    #  Sets the distance type to use throughout FLANN.
    #  If distance type specified is MINKOWSKI, the second argument
    #  specifies which order the minkowski distance should have.
    void flann_set_distance_type(flann_distance_t distance_type, int order);

    #  Gets the distance type in use throughout FLANN.
    flann_distance_t flann_get_distance_type();

    #  Gets the distance order in use throughout FLANN (only applicable if minkowski distance
    #  is in use).
    int flann_get_distance_order();

    #    Builds and returns an index. It uses autotuning if the target_precision field of index_params
    #    is between 0 and 1, or the parameters specified if it's -1.
    #
    #    Params:
    #     dataset = pointer to a data set stored in row major order
    #     rows = number of rows (features) in the dataset
    #     cols = number of columns in the dataset (feature dimensionality)
    #     speedup = speedup over linear search, estimated if using autotuning, output parameter
    #     index_params = index related parameters
    #     flann_params = generic flann parameters
    #
    #    Returns: the newly created index or a number < 0 for error
    flann_index_t flann_build_index(float* dataset,
                                    int rows,
                                    int cols,
                                    float* speedup,
                                    FLANNParameters* flann_params);

    flann_index_t flann_build_index_float(float* dataset,
                                          int rows,
                                          int cols,
                                          float* speedup,
                                          FLANNParameters* flann_params);

    flann_index_t flann_build_index_double(double* dataset,
                                           int rows,
                                           int cols,
                                           float* speedup,
                                           FLANNParameters* flann_params);

    flann_index_t flann_build_index_byte(unsigned char* dataset,
                                         int rows,
                                         int cols,
                                         float* speedup,
                                         FLANNParameters* flann_params);

    flann_index_t flann_build_index_int(int* dataset,
                                        int rows,
                                        int cols,
                                        float* speedup,
                                        FLANNParameters* flann_params);

    #   Adds points to pre-built index.
    #
    #   Params:
    #     index_ptr = pointer to index, must already be built
    #     points = pointer to array of points
    #     rows = number of points to add
    #     columns = feature dimensionality
    #     rebuild_threshold = reallocs index when it grows by factor of
    #       `rebuild_threshold`. A smaller value results is more space efficient
    #       but less computationally efficient. Must be greater than 1.
    #
    #   Returns: 0 if success otherwise -1
    int flann_add_points(flann_index_t index_ptr, float* points,
                                  int rows, int columns,
                                  float rebuild_threshold);

    int flann_add_points_float(flann_index_t index_ptr, float* points,
                                        int rows, int columns,
                                        float rebuild_threshold);

    int flann_add_points_double(flann_index_t index_ptr,
                                         double* points, int rows, int columns,
                                         float rebuild_threshold);

    int flann_add_points_byte(flann_index_t index_ptr,
                                       unsigned char* points, int rows,
                                       int columns, float rebuild_threshold);

    int flann_add_points_int(flann_index_t index_ptr, int* points,
                                      int rows, int columns,
                                      float rebuild_threshold);

    #  Removes a point from a pre-built index.
    #
    #  index_ptr = pointer to pre-built index.
    #  point_id = index of datapoint to remove.
    int flann_remove_point(flann_index_t index_ptr,
                                    unsigned int point_id);

    int flann_remove_point_float(flann_index_t index_ptr,
                                          unsigned int point_id);

    int flann_remove_point_double(flann_index_t index_ptr,
                                           unsigned int point_id);

    int flann_remove_point_byte(flann_index_t index_ptr,
                                         unsigned int point_id);

    int flann_remove_point_int(flann_index_t index_ptr,
                                        unsigned int point_id);

    #  Gets a point from a given index position.
    #
    #  index_ptr = pointer to pre-built index.
    #  point_id = index of datapoint to get.
    #
    #  Returns: pointer to datapoint or NULL on miss
    float* flann_get_point(flann_index_t index_ptr,
                                    unsigned int point_id);

    float* flann_get_point_float(flann_index_t index_ptr,
                                          unsigned int point_id);

    double* flann_get_point_double(flann_index_t index_ptr,
                                            unsigned int point_id);

    unsigned char* flann_get_point_byte(flann_index_t index_ptr,
                                                 unsigned int point_id);

    int* flann_get_point_int(flann_index_t index_ptr,
                                      unsigned int point_id);

    #  Returns the number of datapoints stored in index.
    #
    #  index_ptr = pointer to pre-built index.
    unsigned int flann_veclen(flann_index_t index_ptr);

    unsigned int flann_veclen_float(flann_index_t index_ptr);

    unsigned int flann_veclen_double(flann_index_t index_ptr);

    unsigned int flann_veclen_byte(flann_index_t index_ptr);

    unsigned int flann_veclen_int(flann_index_t index_ptr);

    #  Returns the dimensionality of datapoints stored in index.
    #
    #  index_ptr = pointer to pre-built index.
    unsigned int flann_size(flann_index_t index_ptr);

    unsigned int flann_size_float(flann_index_t index_ptr);

    unsigned int flann_size_double(flann_index_t index_ptr);

    unsigned int flann_size_byte(flann_index_t index_ptr);

    unsigned int flann_size_int(flann_index_t index_ptr);

    #
    #  Returns the number of bytes consumed by the index.
    #
    #  index_ptr = pointer to pre-built index.
    int flann_used_memory(flann_index_t index_ptr);

    int flann_used_memory_float(flann_index_t index_ptr);

    int flann_used_memory_double(flann_index_t index_ptr);

    int flann_used_memory_byte(flann_index_t index_ptr);

    int flann_used_memory_int(flann_index_t index_ptr);

    #  Saves the index to a file. Only the index is saved into the file, the dataset corresponding to the index is not saved.
    #
    #  @param index_id The index that should be saved
    #  @param filename The filename the index should be saved to
    #  @return Returns 0 on success, negative value on error.

    int flann_save_index(flann_index_t index_id,
                                  char* filename);

    int flann_save_index_float(flann_index_t index_id,
                                        char* filename);

    int flann_save_index_double(flann_index_t index_id,
                                         char* filename);

    int flann_save_index_byte(flann_index_t index_id,
                                       char* filename);

    int flann_save_index_int(flann_index_t index_id,
                                      char* filename);

    #  Loads an index from a file.
    #
    #  @param filename File to load the index from.
    #  @param dataset The dataset corresponding to the index.
    #  @param rows Dataset tors
    #  @param cols Dataset columns
    #  @return
    flann_index_t flann_load_index(char* filename,
                                   float* dataset,
                                   int rows,
                                   int cols);

    flann_index_t flann_load_index_float(char* filename,
                                         float* dataset,
                                                  int rows,
                                                  int cols);

    flann_index_t flann_load_index_double(char* filename,
                                                   double* dataset,
                                                   int rows,
                                                   int cols);

    flann_index_t flann_load_index_byte(char* filename,
                                                 unsigned char* dataset,
                                                 int rows,
                                                 int cols);

    flann_index_t flann_load_index_int(char* filename,
                                                int* dataset,
                                                int rows,
                                                int cols);


    #    Builds an index and uses it to find nearest neighbors.
    #
    #    Params:
    #     dataset = pointer to a data set stored in row major order
    #     rows = number of rows (features) in the dataset
    #     cols = number of columns in the dataset (feature dimensionality)
    #     testset = pointer to a query set stored in row major order
    #     trows = number of rows (features) in the query dataset (same dimensionality as features in the dataset)
    #     indices = pointer to matrix for the indices of the nearest neighbors of the testset features in the dataset
    #             (must have trows number of rows and nn number of columns)
    #     nn = how many nearest neighbors to return
    #     flann_params = generic flann parameters
    #
    #    Returns: zero or -1 for error
    int flann_find_nearest_neighbors(float* dataset,
                                              int rows,
                                              int cols,
                                              float* testset,
                                              int trows,
                                              int* indices,
                                              float* dists,
                                              int nn,
                                              FLANNParameters* flann_params);

    int flann_find_nearest_neighbors_float(float* dataset,
                                                    int rows,
                                                    int cols,
                                                    float* testset,
                                                    int trows,
                                                    int* indices,
                                                    float* dists,
                                                    int nn,
                                                    FLANNParameters* flann_params);

    int flann_find_nearest_neighbors_double(double* dataset,
                                                     int rows,
                                                     int cols,
                                                     double* testset,
                                                     int trows,
                                                     int* indices,
                                                     double* dists,
                                                     int nn,
                                                     FLANNParameters* flann_params);

    int flann_find_nearest_neighbors_byte(unsigned char* dataset,
                                                   int rows,
                                                   int cols,
                                                   unsigned char* testset,
                                                   int trows,
                                                   int* indices,
                                                   float* dists,
                                                   int nn,
                                                   FLANNParameters* flann_params);

    int flann_find_nearest_neighbors_int(int* dataset,
                                                  int rows,
                                                  int cols,
                                                  int* testset,
                                                  int trows,
                                                  int* indices,
                                                  float* dists,
                                                  int nn,
                                                  FLANNParameters* flann_params);


    #    Searches for nearest neighbors using the index provided
    #
    #    Params:
    #     index_id = the index (constructed previously using flann_build_index).
    #     testset = pointer to a query set stored in row major order
    #     trows = number of rows (features) in the query dataset (same dimensionality as features in the dataset)
    #     indices = pointer to matrix for the indices of the nearest neighbors of the testset features in the dataset
    #             (must have trows number of rows and nn number of columns)
    #     dists = pointer to matrix for the distances of the nearest neighbors of the testset features in the dataset
    #             (must have trows number of rows and 1 column)
    #     nn = how many nearest neighbors to return
    #     flann_params = generic flann parameters
    #
    #    Returns: zero or a number <0 for error

    int flann_find_nearest_neighbors_index(flann_index_t index_id,
                                                    float* testset,
                                                    int trows,
                                                    int* indices,
                                                    float* dists,
                                                    int nn,
                                                    FLANNParameters* flann_params);

    int flann_find_nearest_neighbors_index_float(flann_index_t index_id,
                                                          float* testset,
                                                          int trows,
                                                          int* indices,
                                                          float* dists,
                                                          int nn,
                                                          FLANNParameters* flann_params);

    int flann_find_nearest_neighbors_index_double(flann_index_t index_id,
                                                           double* testset,
                                                           int trows,
                                                           int* indices,
                                                           double* dists,
                                                           int nn,
                                                           FLANNParameters* flann_params);

    int flann_find_nearest_neighbors_index_byte(flann_index_t index_id,
                                                         unsigned char* testset,
                                                         int trows,
                                                         int* indices,
                                                         float* dists,
                                                         int nn,
                                                         FLANNParameters* flann_params);

    int flann_find_nearest_neighbors_index_int(flann_index_t index_id,
                                                        int* testset,
                                                        int trows,
                                                        int* indices,
                                                        float* dists,
                                                        int nn,
                                                        FLANNParameters* flann_params);

    #  Performs an radius search using an already constructed index.
    #  
    #  In case of radius search, instead of always returning a predetermined
    #  number of nearest neighbours (for example the 10 nearest neighbours), the
    #  search will return all the neighbours found within a search radius
    #  of the query point.
    #  
    #  The check parameter in the FLANNParameters below sets the level of approximation
    #  for the search by only visiting "checks" number of features in the index
    #  (the same way as for the KNN search). A lower value for checks will give
    #  a higher search speedup at the cost of potentially not returning all the
    #  neighbours in the specified radius.
    
    int flann_radius_search(flann_index_t index_ptr, # the index
                            float* query, # query point
                            int* indices, # array for storing the indices found (will be modified)
                            float* dists, # similar, but for storing distances
                            int max_nn,  # size of arrays indices and dists
                            float radius, # search radius (squared radius for euclidian metric)
                            FLANNParameters* flann_params);
    
    int flann_radius_search_float(flann_index_t index_ptr, # the index
                                           float* query, # query point
                                           int* indices, # array for storing the indices found (will be modified)
                                           float* dists, # similar, but for storing distances
                                           int max_nn,  # size of arrays indices and dists
                                           float radius, # search radius (squared radius for euclidian metric)
                                           FLANNParameters* flann_params);
    
    int flann_radius_search_double(flann_index_t index_ptr, # the index
                                            double* query, # query point
                                            int* indices, # array for storing the indices found (will be modified)
                                            double* dists, # similar, but for storing distances
                                            int max_nn,  # size of arrays indices and dists
                                            float radius, # search radius (squared radius for euclidian metric)
                                            FLANNParameters* flann_params);
    
    int flann_radius_search_byte(flann_index_t index_ptr, # the index
                                          unsigned char* query, # query point
                                          int* indices, # array for storing the indices found (will be modified)
                                          float* dists, # similar, but for storing distances
                                          int max_nn,  # size of arrays indices and dists
                                          float radius, # search radius (squared radius for euclidian metric)
                                          FLANNParameters* flann_params);
    
    int flann_radius_search_int(flann_index_t index_ptr, # the index
                                         int* query, # query point
                                         int* indices, # array for storing the indices found (will be modified)
                                         float* dists, # similar, but for storing distances
                                         int max_nn,  # size of arrays indices and dists
                                         float radius, # search radius (squared radius for euclidian metric)
                                         FLANNParameters* flann_params);

    #    Deletes an index and releases the memory used by it.
    #
    #    Params:
    #     index_id = the index (constructed previously using flann_build_index).
    #     flann_params = generic flann parameters
    #
    #    Returns: zero or a number <0 for error
    int flann_free_index(flann_index_t index_id,
                                   FLANNParameters* flann_params);

    int flann_free_index_float(flann_index_t index_id,
                                         FLANNParameters* flann_params);

    int flann_free_index_double(flann_index_t index_id,
                                          FLANNParameters* flann_params);

    int flann_free_index_byte(flann_index_t index_id,
                                        FLANNParameters* flann_params);

    int flann_free_index_int(flann_index_t index_id,
                                       FLANNParameters* flann_params);

    #    Clusters the features in the dataset using a hierarchical kmeans clustering approach.
    #    This is significantly faster than using a flat kmeans clustering for a large number
    #    of clusters.
    #
    #    Params:
    #     dataset = pointer to a data set stored in row major order
    #     rows = number of rows (features) in the dataset
    #     cols = number of columns in the dataset (feature dimensionality)
    #     clusters = number of cluster to compute
    #     result = memory buffer where the output cluster centers are storred
    #     index_params = used to specify the kmeans tree parameters (branching factor, max number of iterations to use)
    #     flann_params = generic flann parameters
    #
    #    Returns: number of clusters computed or a number <0 for error. This number can be different than the number of clusters requested, due to the
    #     way hierarchical clusters are computed. The number of clusters returned will be the highest number of the form
    #     (branch_size-1)*K+1 smaller than the number of clusters requested.
    int flann_compute_cluster_centers(float* dataset,
                                               int rows,
                                               int cols,
                                               int clusters,
                                               float* result,
                                               FLANNParameters* flann_params);

    int flann_compute_cluster_centers_float(float* dataset,
                                                     int rows,
                                                     int cols,
                                                     int clusters,
                                                     float* result,
                                                     FLANNParameters* flann_params);

    int flann_compute_cluster_centers_double(double* dataset,
                                                      int rows,
                                                      int cols,
                                                      int clusters,
                                                      double* result,
                                                      FLANNParameters* flann_params);

    int flann_compute_cluster_centers_byte(unsigned char* dataset,
                                                    int rows,
                                                    int cols,
                                                    int clusters,
                                                    float* result,
                                                    FLANNParameters* flann_params);

    int flann_compute_cluster_centers_int(int* dataset,
                                                   int rows,
                                                   int cols,
                                                   int clusters,
                                                   float* result,
                                                   FLANNParameters* flann_params);

