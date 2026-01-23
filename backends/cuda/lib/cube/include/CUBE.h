#pragma once


#ifdef DOUBLE_PRECISION
typedef double real_t;
#else
typedef float real_t;
#endif

typedef real_t complex_t[2];

#include "kernels.h"
#include "utl.h"
#include "operations.h"
#include <cuda_runtime_api.h>