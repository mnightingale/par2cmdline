#ifndef __GF16_GPU_DEVICE_METAL_H
#define __GF16_GPU_DEVICE_METAL_H

#include "gpu_device.h"

// Appends every Metal device to `out`. Callers assign the stable `id` field;
// this only fills in the device properties.
void gpu_metal_enumerate(std::vector<GPUDeviceInfo>& out);

#endif // defined(__GF16_GPU_DEVICE_METAL_H)
