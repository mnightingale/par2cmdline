// OpenCL device enumeration.
//
// Thin adapter: PAR2ProcOCL already enumerates devices and reports them as
// GF16OCL_DeviceInfo, which GPUDeviceInfo was modelled on, so this only
// translates the fields. Kept separate from the backend for the same reason as
// the Metal and Vulkan pairs -- listing devices should not depend on the
// compute path building.

#include "gpu_device_opencl.h"
#include "controller_ocl.h"

void gpu_opencl_enumerate(std::vector<GPUDeviceInfo>& out) {
	// Resolves OpenCL.dll and populates the platform list. Every other entry
	// point dereferences pointers it sets, so calling anything before this
	// segfaults rather than failing. Non-zero means no runtime or no platform,
	// which is the normal case on a machine without OpenCL and not an error.
	if(PAR2ProcOCL::load_runtime() != 0) return;

	// -1 selects the default platform. Machines with several OpenCL platforms
	// (an NVIDIA and an Intel runtime, say) will only report the default one;
	// enumerating across platforms would need a second id dimension that
	// GPUDeviceInfo does not carry.
	std::vector<GF16OCL_DeviceInfo> devices;
	try {
		devices = PAR2ProcOCL::getDevices();
	} catch(...) {
		return;
	}

	for(size_t i = 0; i < devices.size(); i++) {
		const GF16OCL_DeviceInfo& d = devices[i];
		GPUDeviceInfo info;
		info.api = GPU_API_OPENCL;
		info.name = d.name;
		info.available = d.available;
		info.supported = d.supported;
		info.memory = d.memory;
		info.maxAllocation = d.maxAllocation;
		info.localMemory = d.localMemory;
		info.unifiedMemory = d.unifiedMemory;
		info.maxWorkGroup = d.maxWorkGroup;
		info.workGroupMultiple = d.workGroupMultiple;
		info.computeUnits = d.computeUnits;
		if(!d.supported)
			info.unsupportedReason = "no usable OpenCL kernel for this device";
		out.push_back(info);
	}
}
