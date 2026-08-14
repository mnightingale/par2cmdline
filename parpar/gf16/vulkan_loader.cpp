#include "vulkan_loader.h"

#ifdef _WIN32
# define WIN32_LEAN_AND_MEAN
# include <windows.h>
#else
# include <dlfcn.h>
#endif

// Names the loader ships under. Windows has exactly one; elsewhere the
// unversioned .so belongs to the SDK and is often absent on a machine that has
// only a driver, so the versioned name is tried first.
static const char* const VULKAN_LIB_NAMES[] = {
#if defined(_WIN32)
	"vulkan-1.dll"
#elif defined(__APPLE__)
	// MoltenVK, if someone has installed it. Metal is the better path on that
	// platform, but there is no reason for this to fail if Vulkan is present.
	"libvulkan.1.dylib", "libvulkan.dylib", "libMoltenVK.dylib"
#else
	"libvulkan.so.1", "libvulkan.so"
#endif
};

static void* lib_open(const char* name) {
#ifdef _WIN32
	return (void*)LoadLibraryA(name);
#else
	return dlopen(name, RTLD_NOW | RTLD_LOCAL);
#endif
}

static void lib_close(void* handle) {
#ifdef _WIN32
	FreeLibrary((HMODULE)handle);
#else
	dlclose(handle);
#endif
}

static void* lib_sym(void* handle, const char* name) {
#ifdef _WIN32
	return (void*)GetProcAddress((HMODULE)handle, name);
#else
	return dlsym(handle, name);
#endif
}

VulkanLib::VulkanLib()
: handle(NULL), getInstanceProcAddr(NULL), enumerateInstanceVersion(NULL) {
#define X(name) name = NULL;
	PARPAR_VK_GLOBAL_FUNCS(X)
#undef X
}

VulkanLib::~VulkanLib() {
	unload();
}

bool VulkanLib::load() {
	if(handle) return true;

	for(size_t i = 0; i < sizeof(VULKAN_LIB_NAMES) / sizeof(*VULKAN_LIB_NAMES); i++) {
		handle = lib_open(VULKAN_LIB_NAMES[i]);
		if(handle) break;
	}
	if(!handle) return false;

	getInstanceProcAddr = (PFN_vkGetInstanceProcAddr)
		lib_sym(handle, "vkGetInstanceProcAddr");
	if(!getInstanceProcAddr) {
		unload();
		return false;
	}

	// Absent on a 1.0 loader. Left NULL rather than treated as a failure;
	// instanceVersion() reports 1.0 in that case.
	enumerateInstanceVersion = (PFN_vkEnumerateInstanceVersion)
		getInstanceProcAddr(VK_NULL_HANDLE, "vkEnumerateInstanceVersion");

	bool ok = true;
#define X(name) \
	name = (PFN_##name)getInstanceProcAddr(VK_NULL_HANDLE, #name); \
	if(!name) ok = false;
	PARPAR_VK_GLOBAL_FUNCS(X)
#undef X

	if(!ok) unload();
	return ok;
}

void VulkanLib::unload() {
	if(handle) {
		lib_close(handle);
		handle = NULL;
	}
	getInstanceProcAddr = NULL;
	enumerateInstanceVersion = NULL;
#define X(name) name = NULL;
	PARPAR_VK_GLOBAL_FUNCS(X)
#undef X
}

uint32_t VulkanLib::instanceVersion() const {
	if(!enumerateInstanceVersion) return VK_API_VERSION_1_0;
	uint32_t version = VK_API_VERSION_1_0;
	if(enumerateInstanceVersion(&version) != VK_SUCCESS)
		return VK_API_VERSION_1_0;
	return version;
}

bool VulkanInstanceApi::load(const VulkanLib& lib, VkInstance instance) {
	bool ok = true;
#define X(name) \
	name = (PFN_##name)lib.getInstanceProcAddr(instance, #name); \
	if(!name) ok = false;
	PARPAR_VK_INSTANCE_FUNCS(X)
#undef X
	return ok;
}

bool VulkanDeviceApi::load(const VulkanInstanceApi& inst, VkDevice device) {
	bool ok = true;
#define X(name) \
	name = (PFN_##name)inst.vkGetDeviceProcAddr(device, #name); \
	if(!name) ok = false;
	PARPAR_VK_DEVICE_FUNCS(X)
#undef X
	return ok;
}
