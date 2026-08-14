#include "gpu_device.h"
#include "controller.h"

#ifdef PARPAR_METAL_SUPPORT
# include "gpu_device_metal.h"
# include "controller_metal.h"
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

IPAR2ProcBackend* gpu_create_backend(int deviceId, size_t sliceSize,
                                     unsigned inputGrouping, int numThreads,
                                     std::string* nameOut) {
	if(deviceId < 0) {
		deviceId = gpu_default_device();
		if(deviceId < 0) return nullptr;
	}

	auto devices = gpu_enumerate_devices();
	if(deviceId >= (int)devices.size()) return nullptr;
	const GPUDeviceInfo& info = devices[deviceId];
	if(!info.available || !info.supported) return nullptr;

	switch(info.api) {
#ifdef PARPAR_METAL_SUPPORT
	case GPU_API_METAL: {
		// Devices are numbered across all APIs, so translate back to this
		// backend's own index before constructing.
		int apiIndex = 0;
		for(int i = 0; i < deviceId; i++)
			if(devices[i].api == GPU_API_METAL) apiIndex++;

		PAR2ProcMetal* be = new PAR2ProcMetal(apiIndex);
		if(!be->isAvailable()) { delete be; return nullptr; }
		be->setSliceSize(sliceSize);
		be->setNumThreads(numThreads);
		if(!be->init(inputGrouping)) { delete be; return nullptr; }
		if(nameOut) *nameOut = info.name + " (" + be->getMethodName() + ")";
		return be;
	}
#endif
	default:
		return nullptr;
	}
}
