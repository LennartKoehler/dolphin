#pragma once


#ifdef DOUBLE_PRECISION
typedef double real_t;
typedef real_t complex_t[2];
#else
typedef float real_t;
typedef real_t complex_t[2];
#endif

#include "kernels.h"
#include "utl.h"
#include "operations.h"
#include <cuda_runtime_api.h>