#include "gpu_device.h"

#ifdef PARPAR_METAL_SUPPORT
# include "gpu_device_metal.h"
#endif
#ifdef PARPAR_VULKAN_SUPPORT
# include "gpu_device_vulkan.h"
#endif

std::vector<GPUDeviceInfo> gpu_enumerate_devices() {
	std::vector<GPUDeviceInfo> devices;

#ifdef PARPAR_METAL_SUPPORT
	gpu_metal_enumerate(devices);
#endif
#ifdef PARPAR_VULKAN_SUPPORT
	gpu_vulkan_enumerate(devices);
#endif

	// Assign stable ids after all backends have reported, so --gpu=<id> refers
	// to a position in this combined list rather than a per-API index.
	for(size_t i = 0; i < devices.size(); i++)
		devices[i].id = (int)i;

	return devices;
}

int gpu_default_device() {
	auto devices = gpu_enumerate_devices();

	// Prefer the supported device with the most memory. On systems with both an
	// integrated and a discrete GPU this picks the discrete one, which is what
	// we want: this workload is bandwidth-bound and the discrete part has far
	// more of it.
	int best = -1;
	uint64_t bestMemory = 0;
	for(const auto& d : devices) {
		if(!d.available || !d.supported) continue;
		if(best < 0 || d.memory > bestMemory) {
			best = d.id;
			bestMemory = d.memory;
		}
	}
	return best;
}
