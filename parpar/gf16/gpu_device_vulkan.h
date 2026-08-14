#ifndef __GF16_GPU_DEVICE_VULKAN_H
#define __GF16_GPU_DEVICE_VULKAN_H

#include "gpu_device.h"

// Appends every Vulkan physical device, in enumeration order. Reports nothing
// when no loader or driver is present; that is not an error.
void gpu_vulkan_enumerate(std::vector<GPUDeviceInfo>& out);

#endif
