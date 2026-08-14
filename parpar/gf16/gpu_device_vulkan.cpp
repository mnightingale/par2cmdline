// Vulkan device enumeration.
//
// Kept separate from the compute backend, like the Metal pair: listing devices
// creates an instance and reads properties, nothing more. No shader module is
// built here, so --list-gpus still works on a build whose kernel failed to
// compile, and costs nothing on a machine with no Vulkan at all.

#include "gpu_device_vulkan.h"
#include "vulkan_loader.h"

#include <cstring>

// The kernel stages one output group's lookup tables into shared memory. A
// device offering less than this cannot run it at any group width. Matches the
// Metal backend's threshold.
static const uint64_t VULKAN_MIN_SHARED_MEMORY = 16 * 1024;
static const unsigned VULKAN_MIN_WORKGROUP = 128;

// Assumed when the driver does not report a subgroup size (Vulkan 1.0, where
// VkPhysicalDeviceSubgroupProperties does not exist). 32 is right for NVIDIA
// and Intel; AMD's 64 only means we round the workgroup slightly small.
static const unsigned VULKAN_DEFAULT_SUBGROUP = 32;

namespace {

// Total size of the heaps a device would allocate its working set from.
//
// For a discrete GPU that is the device-local heaps -- VRAM. For an integrated
// one the device-local heap *is* system RAM, and reporting all of it would
// invite the backend to size buffers against memory the host also needs, so it
// is reported as-is and the unifiedMemory flag tells the caller to be careful.
uint64_t device_local_memory(const VkPhysicalDeviceMemoryProperties& mem) {
	uint64_t total = 0;
	for(uint32_t i = 0; i < mem.memoryHeapCount; i++) {
		if(mem.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
			total += mem.memoryHeaps[i].size;
	}
	return total;
}

// True when every device-local heap is also host-visible, i.e. there is one
// physical pool behind both processors. Integrated parts report this; so does
// a discrete card with resizable BAR covering all of VRAM, which is why the
// heap flags are checked rather than just the device type.
bool has_unified_memory(const VkPhysicalDeviceMemoryProperties& mem) {
	bool sawDeviceLocal = false;
	bool allHostVisible = true;
	for(uint32_t i = 0; i < mem.memoryTypeCount; i++) {
		const VkMemoryPropertyFlags f = mem.memoryTypes[i].propertyFlags;
		if(!(f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) continue;
		sawDeviceLocal = true;
		if(!(f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) allHostVisible = false;
	}
	return sawDeviceLocal && allHostVisible;
}

bool has_compute_queue(const VulkanInstanceApi& api, VkPhysicalDevice dev) {
	uint32_t count = 0;
	api.vkGetPhysicalDeviceQueueFamilyProperties(dev, &count, NULL);
	if(!count) return false;

	std::vector<VkQueueFamilyProperties> families(count);
	api.vkGetPhysicalDeviceQueueFamilyProperties(dev, &count, families.data());
	for(uint32_t i = 0; i < count; i++) {
		if(families[i].queueCount && (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT))
			return true;
	}
	return false;
}

} // namespace

void gpu_vulkan_enumerate(std::vector<GPUDeviceInfo>& out) {
	VulkanLib lib;
	if(!lib.load()) return;

	// Ask for 1.1 only when the loader supports it. Naming a version the
	// loader predates makes vkCreateInstance fail outright, so a 1.0 machine
	// would lose the backend entirely rather than losing subgroup reporting.
	const uint32_t loaderVersion = lib.instanceVersion();
	const bool have11 = VK_API_VERSION_MAJOR(loaderVersion) > 1
		|| (VK_API_VERSION_MAJOR(loaderVersion) == 1
		    && VK_API_VERSION_MINOR(loaderVersion) >= 1);

	VkApplicationInfo app = {};
	app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
	app.pApplicationName = "par2";
	app.apiVersion = have11 ? VK_API_VERSION_1_1 : VK_API_VERSION_1_0;

	VkInstanceCreateInfo ci = {};
	ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
	ci.pApplicationInfo = &app;

	VkInstance instance = VK_NULL_HANDLE;
	if(lib.vkCreateInstance(&ci, NULL, &instance) != VK_SUCCESS) return;

	VulkanInstanceApi api;
	if(!api.load(lib, instance)) {
		// No vkDestroyInstance to call it with if that is what failed to
		// resolve, so check before leaking deliberately.
		if(api.vkDestroyInstance) api.vkDestroyInstance(instance, NULL);
		return;
	}

	uint32_t count = 0;
	if(api.vkEnumeratePhysicalDevices(instance, &count, NULL) != VK_SUCCESS || !count) {
		api.vkDestroyInstance(instance, NULL);
		return;
	}
	std::vector<VkPhysicalDevice> devices(count);
	if(api.vkEnumeratePhysicalDevices(instance, &count, devices.data()) != VK_SUCCESS) {
		api.vkDestroyInstance(instance, NULL);
		return;
	}

	for(uint32_t i = 0; i < count; i++) {
		VkPhysicalDeviceProperties props;
		api.vkGetPhysicalDeviceProperties(devices[i], &props);

		VkPhysicalDeviceMemoryProperties mem;
		api.vkGetPhysicalDeviceMemoryProperties(devices[i], &mem);

		GPUDeviceInfo info;
		info.api = GPU_API_VULKAN;
		info.name = props.deviceName;
		info.available = true;

		info.memory = device_local_memory(mem);
		// The binding limit, not the allocation limit: a storage buffer
		// descriptor cannot address more than this however much was allocated,
		// so it is what actually bounds a slice buffer.
		info.maxAllocation = props.limits.maxStorageBufferRange;
		info.localMemory = props.limits.maxComputeSharedMemorySize;
		info.unifiedMemory = has_unified_memory(mem);
		info.maxWorkGroup = props.limits.maxComputeWorkGroupInvocations;
		info.workGroupMultiple = VULKAN_DEFAULT_SUBGROUP;
		// Core Vulkan exposes no compute-unit count. 0 is honest; the backend
		// sizes its dispatch from the workgroup limits instead.
		info.computeUnits = 0;

		if(props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU) {
			// A software rasteriser (lavapipe, SwiftShader). It would run the
			// kernel correctly and far slower than the CPU backend that is
			// already available, and gpu_default_device() picks on memory
			// size, which such a device reports generously.
			info.supported = false;
			info.unsupportedReason = "software renderer";
		} else if(!has_compute_queue(api, devices[i])) {
			info.supported = false;
			info.unsupportedReason = "no compute queue";
		} else if(info.localMemory < VULKAN_MIN_SHARED_MEMORY) {
			info.supported = false;
			info.unsupportedReason = "insufficient shared memory";
		} else if(info.maxWorkGroup < VULKAN_MIN_WORKGROUP) {
			info.supported = false;
			info.unsupportedReason = "workgroup size too small";
		} else {
			info.supported = true;
		}

		out.push_back(info);
	}

	api.vkDestroyInstance(instance, NULL);
}
