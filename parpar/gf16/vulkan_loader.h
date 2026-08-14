#ifndef __GF16_VULKAN_LOADER_H
#define __GF16_VULKAN_LOADER_H

// Runtime Vulkan loader.
//
// Vulkan is resolved with LoadLibrary/dlopen rather than linked, so a binary
// built with the Vulkan backend still starts on a machine that has no Vulkan
// driver at all -- gpu_enumerate_devices() simply reports nothing and the CPU
// backend is used. Linking would make the loader a hard runtime dependency of
// par2 itself, which is exactly what the GPU backend is supposed to avoid.
//
// VK_NO_PROTOTYPES suppresses the declarations that would otherwise pull in
// link-time symbols; every entry point below is a function pointer.
//
// Entry points come in three tiers, and using the wrong tier is a real bug
// rather than a style question:
//
//   global    resolved from the library with a NULL instance
//   instance  resolved once an instance exists; dispatch on it is generic
//   device    resolved from the device, which lets the loader skip its own
//             dispatch layer -- this matters on the hot path
//
// The X-macro lists keep the pointer declarations, the resolution loop and the
// "did everything resolve" check in agreement; adding a function in one place
// is enough.

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

// Resolved with a NULL instance.
#define PARPAR_VK_GLOBAL_FUNCS(X) \
	X(vkCreateInstance)

// Resolved from the instance. vkGetDeviceProcAddr is here because it is how
// the device tier is populated.
#define PARPAR_VK_INSTANCE_FUNCS(X) \
	X(vkDestroyInstance) \
	X(vkEnumeratePhysicalDevices) \
	X(vkGetPhysicalDeviceProperties) \
	X(vkGetPhysicalDeviceMemoryProperties) \
	X(vkGetPhysicalDeviceQueueFamilyProperties) \
	X(vkCreateDevice) \
	X(vkDestroyDevice) \
	X(vkGetDeviceProcAddr)

// Resolved from the device.
#define PARPAR_VK_DEVICE_FUNCS(X) \
	X(vkGetDeviceQueue) \
	X(vkCreateBuffer) \
	X(vkDestroyBuffer) \
	X(vkGetBufferMemoryRequirements) \
	X(vkAllocateMemory) \
	X(vkFreeMemory) \
	X(vkBindBufferMemory) \
	X(vkMapMemory) \
	X(vkUnmapMemory) \
	X(vkFlushMappedMemoryRanges) \
	X(vkInvalidateMappedMemoryRanges) \
	X(vkCreateShaderModule) \
	X(vkDestroyShaderModule) \
	X(vkCreateDescriptorSetLayout) \
	X(vkDestroyDescriptorSetLayout) \
	X(vkCreateDescriptorPool) \
	X(vkDestroyDescriptorPool) \
	X(vkAllocateDescriptorSets) \
	X(vkUpdateDescriptorSets) \
	X(vkCreatePipelineLayout) \
	X(vkDestroyPipelineLayout) \
	X(vkCreateComputePipelines) \
	X(vkDestroyPipeline) \
	X(vkCreateCommandPool) \
	X(vkDestroyCommandPool) \
	X(vkAllocateCommandBuffers) \
	X(vkFreeCommandBuffers) \
	X(vkBeginCommandBuffer) \
	X(vkEndCommandBuffer) \
	X(vkResetCommandBuffer) \
	X(vkCmdBindPipeline) \
	X(vkCmdBindDescriptorSets) \
	X(vkCmdPushConstants) \
	X(vkCmdDispatch) \
	X(vkCmdCopyBuffer) \
	X(vkCmdPipelineBarrier) \
	X(vkQueueSubmit) \
	X(vkQueueWaitIdle) \
	X(vkDeviceWaitIdle) \
	X(vkCreateFence) \
	X(vkDestroyFence) \
	X(vkResetFences) \
	X(vkWaitForFences) \
	X(vkGetFenceStatus)

// The library plus the entry points reachable without an instance.
struct VulkanLib {
	void* handle;
	PFN_vkGetInstanceProcAddr getInstanceProcAddr;
	// Vulkan 1.1+; NULL on a 1.0 loader, which is not an error.
	PFN_vkEnumerateInstanceVersion enumerateInstanceVersion;
#define X(name) PFN_##name name;
	PARPAR_VK_GLOBAL_FUNCS(X)
#undef X

	VulkanLib();
	~VulkanLib();

	// False when no Vulkan library is present or it is too old to use. Never
	// an error: the caller falls back to the CPU backend.
	bool load();
	void unload();

	// Highest instance API version the loader supports, as VK_MAKE_VERSION.
	uint32_t instanceVersion() const;

private:
	VulkanLib(const VulkanLib&);
	VulkanLib& operator=(const VulkanLib&);
};

struct VulkanInstanceApi {
#define X(name) PFN_##name name;
	PARPAR_VK_INSTANCE_FUNCS(X)
#undef X

	// False if any entry point is missing, which means the driver is not
	// usable rather than that something went wrong.
	bool load(const VulkanLib& lib, VkInstance instance);
};

struct VulkanDeviceApi {
#define X(name) PFN_##name name;
	PARPAR_VK_DEVICE_FUNCS(X)
#undef X

	bool load(const VulkanInstanceApi& inst, VkDevice device);
};

#endif // defined(__GF16_VULKAN_LOADER_H)
