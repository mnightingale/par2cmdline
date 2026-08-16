#include "gpu_device.h"
#include <cstdlib>
#include "controller.h"

#ifdef PARPAR_METAL_SUPPORT
# include "gpu_device_metal.h"
# include "controller_metal.h"
#endif
#ifdef PARPAR_VULKAN_SUPPORT
# include "gpu_device_vulkan.h"
# include "controller_vulkan.h"
#endif
#ifdef PARPAR_OPENCL_SUPPORT
# include "gpu_device_opencl.h"
# include "controller_ocl.h"
#endif

std::vector<GPUDeviceInfo> gpu_enumerate_devices() {
	std::vector<GPUDeviceInfo> devices;

#ifdef PARPAR_METAL_SUPPORT
	gpu_metal_enumerate(devices);
#endif
#ifdef PARPAR_VULKAN_SUPPORT
	gpu_vulkan_enumerate(devices);
#endif
#ifdef PARPAR_OPENCL_SUPPORT
	// Last, so it does not displace Vulkan as the default pick on a machine
	// where both address the same physical device.
	gpu_opencl_enumerate(devices);
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
	//
	// API rank comes first, though. One physical device is usually reachable
	// through more than one API, and the reported memory differs between them
	// (OpenCL reports a little more of the same card than Vulkan does), so
	// ranking on memory alone would let that reporting difference decide the
	// backend. OpenCL exists here to be measured against the native backends,
	// so it is only chosen when nothing else can drive the device.
	auto rank = [](GPUApi api) -> int {
		switch(api) {
			case GPU_API_METAL:  return 0;
			case GPU_API_VULKAN: return 0;
			case GPU_API_OPENCL: return 1;
			default:             return 2;
		}
	};

	int best = -1, bestRank = 0;
	uint64_t bestMemory = 0;
	for(const auto& d : devices) {
		if(!d.available || !d.supported) continue;
		const int r = rank(d.api);
		if(best < 0 || r < bestRank || (r == bestRank && d.memory > bestMemory)) {
			best = d.id;
			bestRank = r;
			bestMemory = d.memory;
		}
	}
	return best;
}

// Number of staging areas each GPU backend keeps in flight.
//
// Two is enough on unified memory, where staging is a memcpy into memory the
// GPU already sees. On a discrete card it is a real PCIe transfer that has to
// overlap compute, and more batches in flight may be needed to keep the device
// fed - so this is tunable at runtime rather than requiring a rebuild to
// investigate. See docs/gpu-backend.md.
//
// Device memory scales linearly with this: each area holds a full input batch
// plus its lookup tables, so raising it on a card with limited VRAM can push
// allocation past what the device will give back.
#if defined(PARPAR_METAL_SUPPORT) || defined(PARPAR_VULKAN_SUPPORT)
static int gpu_staging_areas() {
	const char* env = getenv("PARPAR_GPU_STAGING");
	if(!env || !*env) return 2;
	int n = atoi(env);
	if(n < 2) n = 2;   // canAdd/addInput assume at least one spare area
	if(n > 8) n = 8;
	return n;
}
#endif

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

		PAR2ProcMetal* be = new PAR2ProcMetal(apiIndex, gpu_staging_areas());
		if(!be->isAvailable()) { delete be; return nullptr; }
		be->setSliceSize(sliceSize);
		be->setNumThreads(numThreads);
		if(!be->init(inputGrouping)) { delete be; return nullptr; }
		if(nameOut) *nameOut = info.name + " (" + be->getMethodName() + ")";
		return be;
	}
#endif
#ifdef PARPAR_VULKAN_SUPPORT
	case GPU_API_VULKAN: {
		int apiIndex = 0;
		for(int i = 0; i < deviceId; i++)
			if(devices[i].api == GPU_API_VULKAN) apiIndex++;

		PAR2ProcVulkan* be = new PAR2ProcVulkan(apiIndex, gpu_staging_areas());
		if(!be->isAvailable()) { delete be; return nullptr; }
		be->setSliceSize(sliceSize);
		be->setNumThreads(numThreads);
		if(!be->init(inputGrouping)) { delete be; return nullptr; }
		if(nameOut) *nameOut = info.name + " (" + be->getMethodName() + ")";
		return be;
	}
#endif
#ifdef PARPAR_OPENCL_SUPPORT
	case GPU_API_OPENCL: {
		int apiIndex = 0;
		for(int i = 0; i < deviceId; i++)
			if(devices[i].api == GPU_API_OPENCL) apiIndex++;

		// Default platform; see the note in gpu_device_opencl.cpp. The backend
		// picks its own kernel and geometry, so inputGrouping is the only hint
		// passed through - it maps to the input batch size, as elsewhere.
		PAR2ProcOCL* be = new PAR2ProcOCL(-1, apiIndex);
		be->setSliceSize(sliceSize);
		if(!be->init(GF16OCL_AUTO, inputGrouping)) { delete be; return nullptr; }
		if(nameOut) *nameOut = info.name + " (OpenCL " + be->getMethodName() + ")";
		return be;
	}
#endif
	default:
		return nullptr;
	}
}
