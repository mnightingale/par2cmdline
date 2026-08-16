#ifndef __GF16_GPU_DEVICE_OPENCL_H
#define __GF16_GPU_DEVICE_OPENCL_H

#include "gpu_device.h"

// Appends every OpenCL device on the default platform. Reports nothing when no
// OpenCL runtime or device is present; that is not an error.
void gpu_opencl_enumerate(std::vector<GPUDeviceInfo>& out);

#endif
