#ifndef __GF16_GPU_DEVICE_H
#define __GF16_GPU_DEVICE_H

// API-agnostic GPU device description and enumeration.
//
// Each GPU backend (Metal, Vulkan) registers its devices here so the front-end
// can list and select them without knowing which API reaches a given device.
// Modelled on GF16OCL_DeviceInfo in controller_ocl.h, minus the OpenCL types.

#include "../src/stdint.h"
#include <string>
#include <vector>

enum GPUApi {
	GPU_API_NONE = 0,
	GPU_API_METAL,
	GPU_API_VULKAN
};

struct GPUDeviceInfo {
	// Stable index used by --gpu=<id>. Unique across all APIs.
	int id;
	GPUApi api;
	std::string name;

	bool available;   // device present and usable
	bool supported;   // we can actually run our kernels on it
	std::string unsupportedReason; // why not, when supported == false

	// Memory
	uint64_t memory;          // total/recommended working set, bytes
	uint64_t maxAllocation;   // largest single buffer, bytes
	uint64_t localMemory;     // threadgroup/shared memory per group, bytes
	bool unifiedMemory;       // host and device share physical memory

	// Execution geometry
	unsigned maxWorkGroup;      // max threads per threadgroup
	unsigned workGroupMultiple; // preferred multiple (warp/wavefront/SIMD width)
	unsigned computeUnits;

	GPUDeviceInfo()
	: id(-1), api(GPU_API_NONE), available(false), supported(false),
	  memory(0), maxAllocation(0), localMemory(0), unifiedMemory(false),
	  maxWorkGroup(0), workGroupMultiple(0), computeUnits(0) {}
};

inline const char* gpu_api_name(GPUApi api) {
	switch(api) {
		case GPU_API_METAL:  return "Metal";
		case GPU_API_VULKAN: return "Vulkan";
		default:             return "none";
	}
}

// Every GPU device across all compiled-in APIs, in stable id order.
// Returns an empty vector when no backend is compiled in or no device exists;
// this is never an error - the CPU backend always remains available.
std::vector<GPUDeviceInfo> gpu_enumerate_devices();

// Index into gpu_enumerate_devices() of the device we would pick for
// --gpu=auto, or -1 if none is suitable.
int gpu_default_device();

#endif // defined(__GF16_GPU_DEVICE_H)
