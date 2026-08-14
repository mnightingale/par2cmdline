// Metal device enumeration.
//
// Kept separate from the compute backend so that listing devices costs nothing
// beyond creating MTLDevice handles - no shader library is loaded here, which
// means --list-gpus works even on a build whose kernels failed to compile.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "gpu_device_metal.h"

// The GF16 kernels build per-coefficient lookup tables in threadgroup memory.
// A device offering less than this cannot run them.
static const uint64_t METAL_MIN_THREADGROUP_MEMORY = 16 * 1024;
static const unsigned METAL_MIN_WORKGROUP = 128;

// Apple GPUs execute in SIMD groups of 32; Metal exposes this per-pipeline
// rather than per-device, so we assume the architectural value here and let the
// compute backend refine it from the real pipeline state once one exists.
static const unsigned METAL_SIMD_WIDTH = 32;

void gpu_metal_enumerate(std::vector<GPUDeviceInfo>& out) {
	@autoreleasepool {
		NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
		if(!devices) return;

		for(id<MTLDevice> dev in devices) {
			GPUDeviceInfo info;
			info.api = GPU_API_METAL;
			info.name = [[dev name] UTF8String];
			info.available = true;

			info.memory = [dev recommendedMaxWorkingSetSize];
			info.maxAllocation = [dev maxBufferLength];
			info.localMemory = [dev maxThreadgroupMemoryLength];
			info.unifiedMemory = [dev hasUnifiedMemory];
			info.maxWorkGroup = (unsigned)[dev maxThreadsPerThreadgroup].width;
			info.workGroupMultiple = METAL_SIMD_WIDTH;

			// Metal exposes no compute-unit count. Leaving it at 0 is honest;
			// the backend sizes its dispatch from threadgroup limits instead.
			info.computeUnits = 0;

			if(info.localMemory < METAL_MIN_THREADGROUP_MEMORY) {
				info.supported = false;
				info.unsupportedReason = "insufficient threadgroup memory";
			} else if(info.maxWorkGroup < METAL_MIN_WORKGROUP) {
				info.supported = false;
				info.unsupportedReason = "threadgroup size too small";
			} else {
				info.supported = true;
			}

			out.push_back(info);
		}
	}
}
