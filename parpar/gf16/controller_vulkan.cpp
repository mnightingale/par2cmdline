#include "controller_vulkan.h"
#include "vulkan_loader.h"
#include "gfmat_coeff.h"

#include <cassert>
#include <cstring>
#include <algorithm>
#include <thread>
#include <chrono>
#include <cstdlib>
#include <cstdio>

// Compiled from gf16_vulkan.comp by the build; defines gf16_vulkan_spv[].
#include "gf16_vulkan_spv.h"

// Set PARPAR_GPU_STATS=1 to print a breakdown of where GPU-path time goes.
// Answers the question a wall-clock number cannot: whether a disappointing
// result is the kernel, the host-side staging, or lost overlap between them.
static bool gpu_stats_enabled() {
	static const bool enabled = getenv("PARPAR_GPU_STATS") != NULL;
	return enabled;
}

namespace {
struct GpuStats {
	std::atomic<uint64_t> stageNs{0}, stageBytes{0};
	std::atomic<uint64_t> lutNs{0};
	std::atomic<uint64_t> readbackNs{0};
	std::atomic<uint64_t> gpuNs{0};
	std::atomic<uint64_t> encodeNs{0};
	std::atomic<uint64_t> dispatches{0};
	double wallStart = 0;
};
GpuStats g_stats;

inline double now_s() {
	using namespace std::chrono;
	return duration<double>(steady_clock::now().time_since_epoch()).count();
}
inline uint64_t now_ns() {
	using namespace std::chrono;
	return (uint64_t)duration_cast<nanoseconds>(
		steady_clock::now().time_since_epoch()).count();
}
}

// Must match the kernel's definitions.
#define GF16_LUT_ENTRIES 64
#define GF16_MAX_OUTPUTS_PER_GROUP 8
#define GF16_VECTOR_BYTES 16 // uvec4

// The kernel reads 32-bit table entries; see the note in gf16_vulkan.comp.
typedef uint32_t gf16_lut_entry;

struct GF16Params {
	uint32_t numInputs;
	uint32_t numOutputs;
	uint32_t outputsPerGroup;
	uint32_t vecPerSlice;
	uint32_t sliceStrideVec;
	uint32_t accumulate;
};

namespace {

struct VkBufferAlloc {
	VkBuffer buffer;
	VkDeviceMemory memory;
	void* mapped;
	VkDeviceSize size;
	VkBufferAlloc() : buffer(VK_NULL_HANDLE), memory(VK_NULL_HANDLE),
	                  mapped(NULL), size(0) {}
};

// One staging area's device-side resources. hostX is written by the transfer
// threads; devX is what the kernel actually reads.
struct VulkanArea {
	VkBufferAlloc hostInput, devInput;
	VkBufferAlloc hostLut, devLut;
	VkDescriptorSet set;
	VkCommandBuffer cmd;
	VkFence fence;
	VkQueryPool queries;
	VulkanArea() : set(VK_NULL_HANDLE), cmd(VK_NULL_HANDLE),
	               fence(VK_NULL_HANDLE), queries(VK_NULL_HANDLE) {}
};

// Per transfer thread, so readbacks do not contend for one staging buffer or
// one command pool (VkCommandPool is externally synchronised).
struct VulkanReadback {
	VkBufferAlloc host;
	VkCommandPool pool;
	VkCommandBuffer cmd;
	VkFence fence;
	VulkanReadback() : pool(VK_NULL_HANDLE), cmd(VK_NULL_HANDLE),
	                   fence(VK_NULL_HANDLE) {}
};

} // namespace

struct PAR2ProcVulkanImpl {
	VulkanLib lib;
	VulkanInstanceApi inst;
	VulkanDeviceApi dev;

	VkInstance instance;
	VkPhysicalDevice physical;
	VkDevice device;
	VkQueue queue;
	uint32_t queueFamily;
	bool haveTimestamps;
	float timestampPeriod;

	VkPhysicalDeviceProperties props;
	VkPhysicalDeviceMemoryProperties memProps;

	VkShaderModule shader;
	VkDescriptorSetLayout setLayout;
	VkPipelineLayout pipeLayout;
	VkPipeline pipeline;
	VkDescriptorPool descPool;
	VkCommandPool cmdPool;

	// VkQueue is externally synchronised, and both dispatch and readback submit
	// to it from different threads.
	std::mutex queueMutex;

	std::vector<VulkanArea> areas;
	VkBufferAlloc devOutput;
	std::vector<VulkanReadback> readback;

	std::string deviceName;

	PAR2ProcVulkanImpl()
	: instance(VK_NULL_HANDLE), physical(VK_NULL_HANDLE), device(VK_NULL_HANDLE),
	  queue(VK_NULL_HANDLE), queueFamily(0), haveTimestamps(false),
	  timestampPeriod(1.0f), shader(VK_NULL_HANDLE), setLayout(VK_NULL_HANDLE),
	  pipeLayout(VK_NULL_HANDLE), pipeline(VK_NULL_HANDLE),
	  descPool(VK_NULL_HANDLE), cmdPool(VK_NULL_HANDLE) {
		std::memset(&props, 0, sizeof(props));
		std::memset(&memProps, 0, sizeof(memProps));
	}

	uint32_t findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags want) const {
		for(uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
			if(!(typeBits & (1u << i))) continue;
			if((memProps.memoryTypes[i].propertyFlags & want) == want) return i;
		}
		return UINT32_MAX;
	}

	bool createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
	                  VkMemoryPropertyFlags want, VkBufferAlloc& out, bool map) {
		destroyBuffer(out);
		if(size == 0) return true;

		VkBufferCreateInfo bi = {};
		bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bi.size = size;
		bi.usage = usage;
		bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		if(dev.vkCreateBuffer(device, &bi, NULL, &out.buffer) != VK_SUCCESS)
			return false;

		VkMemoryRequirements req;
		dev.vkGetBufferMemoryRequirements(device, out.buffer, &req);

		uint32_t type = findMemoryType(req.memoryTypeBits, want);
		if(type == UINT32_MAX) {
			// Fall back to a weaker match rather than failing outright: a
			// device-local request on an integrated part may only be
			// satisfiable by host-visible memory, which still works.
			if(want & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
				type = findMemoryType(req.memoryTypeBits,
					want & ~(VkMemoryPropertyFlags)VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
			if(type == UINT32_MAX) { destroyBuffer(out); return false; }
		}

		VkMemoryAllocateInfo ai = {};
		ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		ai.allocationSize = req.size;
		ai.memoryTypeIndex = type;
		if(dev.vkAllocateMemory(device, &ai, NULL, &out.memory) != VK_SUCCESS) {
			destroyBuffer(out);
			return false;
		}
		if(dev.vkBindBufferMemory(device, out.buffer, out.memory, 0) != VK_SUCCESS) {
			destroyBuffer(out);
			return false;
		}
		if(map) {
			// Mapped for the buffer's whole lifetime. Vulkan permits this, and
			// mapping per slice would cost more than the copy itself.
			if(dev.vkMapMemory(device, out.memory, 0, VK_WHOLE_SIZE, 0,
			                   &out.mapped) != VK_SUCCESS) {
				destroyBuffer(out);
				return false;
			}
		}
		out.size = size;
		return true;
	}

	void destroyBuffer(VkBufferAlloc& b) {
		if(device == VK_NULL_HANDLE) { b = VkBufferAlloc(); return; }
		if(b.mapped) { dev.vkUnmapMemory(device, b.memory); b.mapped = NULL; }
		if(b.buffer) { dev.vkDestroyBuffer(device, b.buffer, NULL); b.buffer = VK_NULL_HANDLE; }
		if(b.memory) { dev.vkFreeMemory(device, b.memory, NULL); b.memory = VK_NULL_HANDLE; }
		b.size = 0;
	}

	// Total size of the device-local heaps, i.e. the VRAM budget to size
	// buffers against.
	uint64_t deviceLocalMemory() const {
		uint64_t total = 0;
		for(uint32_t i = 0; i < memProps.memoryHeapCount; i++) {
			if(memProps.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
				total += memProps.memoryHeaps[i].size;
		}
		return total;
	}
};

// ---- GF(2^16) helpers for lookup-table construction ----------------------
//
// Same derivation as controller_metal.mm; only the entry width differs.

static const uint32_t GF16_POLY = 0x1100b;

static inline uint16_t gf16_double(uint16_t v) {
	uint32_t x = (uint32_t)v << 1;
	if(x & 0x10000) x ^= GF16_POLY;
	return (uint16_t)x;
}

// Four 16-entry tables for coefficient c: L[j*16 + n] = c * (n << 4j).
static void gf16_build_lut(gf16_lut_entry* L, uint16_t c) {
	uint16_t t[GF16_LUT_ENTRIES];
	t[0] = 0;
	t[1] = c;
	t[2] = gf16_double(t[1]);
	t[4] = gf16_double(t[2]);
	t[8] = gf16_double(t[4]);
	for(unsigned n = 3; n < 16; n++) {
		if(n == 4 || n == 8) continue; // already a power of two
		t[n] = t[n & (n - 1)] ^ t[n & (unsigned)(-(int)n)];
	}
	for(unsigned j = 1; j < 4; j++) {
		uint16_t* prev = t + (j - 1) * 16;
		uint16_t* cur = t + j * 16;
		for(unsigned n = 0; n < 16; n++)
			cur[n] = gf16_double(gf16_double(gf16_double(gf16_double(prev[n]))));
	}
	for(unsigned n = 0; n < GF16_LUT_ENTRIES; n++)
		L[n] = t[n];
}

// ---- construction --------------------------------------------------------

PAR2ProcVulkan::PAR2ProcVulkan(int _deviceId, int stagingAreas)
: IPAR2ProcBackend(), impl(new PAR2ProcVulkanImpl()), initSuccess(false),
  deviceId(_deviceId), sliceSize(0), sliceSizeCksum(0), sliceSizeAligned(0),
  allocatedSliceSize(0), outputsPerGroup(GF16_MAX_OUTPUTS_PER_GROUP),
  threadsPerGroup(256), gf(nullptr), gfMethod(GF16_AUTO), staging(stagingAreas),
  nextTransferThread(0), numThreads(0)
{
	if(!impl->lib.load()) return;

	const uint32_t loaderVersion = impl->lib.instanceVersion();
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
	if(impl->lib.vkCreateInstance(&ci, NULL, &impl->instance) != VK_SUCCESS) return;
	if(!impl->inst.load(impl->lib, impl->instance)) return;

	uint32_t count = 0;
	if(impl->inst.vkEnumeratePhysicalDevices(impl->instance, &count, NULL) != VK_SUCCESS
	   || !count) return;
	std::vector<VkPhysicalDevice> devices(count);
	if(impl->inst.vkEnumeratePhysicalDevices(impl->instance, &count,
	                                         devices.data()) != VK_SUCCESS) return;

	const uint32_t idx = (_deviceId < 0) ? 0 : (uint32_t)_deviceId;
	if(idx >= count) return;
	impl->physical = devices[idx];

	impl->inst.vkGetPhysicalDeviceProperties(impl->physical, &impl->props);
	impl->inst.vkGetPhysicalDeviceMemoryProperties(impl->physical, &impl->memProps);
	impl->deviceName = impl->props.deviceName;
	impl->timestampPeriod = impl->props.limits.timestampPeriod;

	// Pick a compute-capable queue family, preferring one without graphics --
	// on discrete hardware that is an async compute engine, which will not be
	// contending with a desktop compositor for the same slots.
	uint32_t famCount = 0;
	impl->inst.vkGetPhysicalDeviceQueueFamilyProperties(impl->physical, &famCount, NULL);
	if(!famCount) return;
	std::vector<VkQueueFamilyProperties> fams(famCount);
	impl->inst.vkGetPhysicalDeviceQueueFamilyProperties(impl->physical, &famCount,
	                                                    fams.data());
	uint32_t chosen = UINT32_MAX;
	for(uint32_t i = 0; i < famCount; i++) {
		if(!fams[i].queueCount || !(fams[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) continue;
		if(chosen == UINT32_MAX) chosen = i;
		if(!(fams[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) { chosen = i; break; }
	}
	if(chosen == UINT32_MAX) return;
	impl->queueFamily = chosen;
	impl->haveTimestamps = fams[chosen].timestampValidBits > 0
		&& impl->props.limits.timestampPeriod > 0;

	const float priority = 1.0f;
	VkDeviceQueueCreateInfo qi = {};
	qi.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
	qi.queueFamilyIndex = chosen;
	qi.queueCount = 1;
	qi.pQueuePriorities = &priority;

	VkDeviceCreateInfo di = {};
	di.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
	di.queueCreateInfoCount = 1;
	di.pQueueCreateInfos = &qi;
	if(impl->inst.vkCreateDevice(impl->physical, &di, NULL, &impl->device) != VK_SUCCESS)
		return;
	if(!impl->dev.load(impl->inst, impl->device)) return;
	impl->dev.vkGetDeviceQueue(impl->device, chosen, 0, &impl->queue);

	VkShaderModuleCreateInfo si = {};
	si.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	si.codeSize = sizeof(gf16_vulkan_spv);
	si.pCode = gf16_vulkan_spv;
	if(impl->dev.vkCreateShaderModule(impl->device, &si, NULL, &impl->shader) != VK_SUCCESS)
		return;

	// Three storage buffers: output, input batch, lookup tables.
	VkDescriptorSetLayoutBinding bindings[3] = {};
	for(unsigned i = 0; i < 3; i++) {
		bindings[i].binding = i;
		bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		bindings[i].descriptorCount = 1;
		bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	}
	VkDescriptorSetLayoutCreateInfo li = {};
	li.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	li.bindingCount = 3;
	li.pBindings = bindings;
	if(impl->dev.vkCreateDescriptorSetLayout(impl->device, &li, NULL,
	                                         &impl->setLayout) != VK_SUCCESS) return;

	VkPushConstantRange pcr = {};
	pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	pcr.offset = 0;
	pcr.size = sizeof(GF16Params);

	VkPipelineLayoutCreateInfo pli = {};
	pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pli.setLayoutCount = 1;
	pli.pSetLayouts = &impl->setLayout;
	pli.pushConstantRangeCount = 1;
	pli.pPushConstantRanges = &pcr;
	if(impl->dev.vkCreatePipelineLayout(impl->device, &pli, NULL,
	                                    &impl->pipeLayout) != VK_SUCCESS) return;

	VkCommandPoolCreateInfo cpi = {};
	cpi.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
	cpi.queueFamilyIndex = chosen;
	if(impl->dev.vkCreateCommandPool(impl->device, &cpi, NULL,
	                                 &impl->cmdPool) != VK_SUCCESS) return;

	impl->areas.resize(staging.size());
	initSuccess = true;
}

PAR2ProcVulkan::~PAR2ProcVulkan() {
	deinit();

	// Order matters: everything below is owned by the device, and the device
	// by the instance.
	if(impl->device != VK_NULL_HANDLE) {
		if(impl->pipeline) impl->dev.vkDestroyPipeline(impl->device, impl->pipeline, NULL);
		if(impl->pipeLayout) impl->dev.vkDestroyPipelineLayout(impl->device, impl->pipeLayout, NULL);
		if(impl->setLayout) impl->dev.vkDestroyDescriptorSetLayout(impl->device, impl->setLayout, NULL);
		if(impl->shader) impl->dev.vkDestroyShaderModule(impl->device, impl->shader, NULL);
		if(impl->descPool) impl->dev.vkDestroyDescriptorPool(impl->device, impl->descPool, NULL);
		if(impl->cmdPool) impl->dev.vkDestroyCommandPool(impl->device, impl->cmdPool, NULL);
		impl->inst.vkDestroyDevice(impl->device, NULL);
		impl->device = VK_NULL_HANDLE;
	}
	if(impl->instance != VK_NULL_HANDLE) {
		impl->inst.vkDestroyInstance(impl->instance, NULL);
		impl->instance = VK_NULL_HANDLE;
	}
}

const char* PAR2ProcVulkan::getMethodName() const {
	return "Vulkan (nibble LUT)";
}

// `areas` is reported so a run's numbers carry the staging depth that produced
// them - the figure PARPAR_GPU_STAGING varies.
static void gpu_stats_report(double wall, unsigned areas) {
	const double s = 1e-9;
	fprintf(stderr,
		"[GPU STATS] wall %.2fs | gpu %.2fs over %llu dispatches | "
		"stage %.2fs (%.1f GiB) | lut+encode %.2fs | readback %.2fs | "
		"staging areas %u\n",
		wall,
		g_stats.gpuNs.load() * s, (unsigned long long)g_stats.dispatches.load(),
		g_stats.stageNs.load() * s,
		g_stats.stageBytes.load() / 1073741824.0,
		g_stats.encodeNs.load() * s,
		g_stats.readbackNs.load() * s,
		areas);
}

void PAR2ProcVulkan::_deinit() {
	drainTransfers();
	if(gpu_stats_enabled() && g_stats.wallStart > 0) {
		gpu_stats_report(now_s() - g_stats.wallStart, getStagingAreas());
		g_stats.wallStart = 0;
	}
	if(impl->device == VK_NULL_HANDLE) return;

	// Everything below frees memory the GPU may still be reading.
	impl->dev.vkDeviceWaitIdle(impl->device);

	for(size_t i = 0; i < impl->areas.size(); i++) {
		VulkanArea& a = impl->areas[i];
		impl->destroyBuffer(a.hostInput);
		impl->destroyBuffer(a.devInput);
		impl->destroyBuffer(a.hostLut);
		impl->destroyBuffer(a.devLut);
		if(a.fence) { impl->dev.vkDestroyFence(impl->device, a.fence, NULL); a.fence = VK_NULL_HANDLE; }
		if(a.queries) { impl->dev.vkDestroyQueryPool(impl->device, a.queries, NULL); a.queries = VK_NULL_HANDLE; }
		if(a.cmd) {
			impl->dev.vkFreeCommandBuffers(impl->device, impl->cmdPool, 1, &a.cmd);
			a.cmd = VK_NULL_HANDLE;
		}
		a.set = VK_NULL_HANDLE; // owned by the pool, freed below
	}
	for(size_t i = 0; i < impl->readback.size(); i++) {
		VulkanReadback& r = impl->readback[i];
		impl->destroyBuffer(r.host);
		if(r.fence) { impl->dev.vkDestroyFence(impl->device, r.fence, NULL); r.fence = VK_NULL_HANDLE; }
		if(r.pool) {
			impl->dev.vkDestroyCommandPool(impl->device, r.pool, NULL);
			r.pool = VK_NULL_HANDLE;
			r.cmd = VK_NULL_HANDLE;
		}
	}
	impl->readback.clear();
	impl->destroyBuffer(impl->devOutput);

	if(impl->descPool) {
		impl->dev.vkDestroyDescriptorPool(impl->device, impl->descPool, NULL);
		impl->descPool = VK_NULL_HANDLE;
	}
	allocatedSliceSize = 0;
}

void PAR2ProcVulkan::freeProcessingMem() {
	if(impl->device == VK_NULL_HANDLE) return;
	impl->dev.vkDeviceWaitIdle(impl->device);
	impl->destroyBuffer(impl->devOutput);
	// Every descriptor set still points at the buffer just freed. Forcing the
	// next setCurrentSliceSize() to reallocate is what stops a subsequent
	// dispatch binding them: reallocBuffers() short-circuits when the slice
	// size has not grown, which would otherwise leave them dangling.
	allocatedSliceSize = 0;
}

void PAR2ProcVulkan::setNumThreads(int threads) {
	numThreads = threads;
}

bool PAR2ProcVulkan::init(unsigned inputGrouping, Galois16Methods cksumMethod) {
	if(!initSuccess) return false;

	// Staging is memcpy-and-checksum bound, so a handful of threads saturates
	// it; more would just contend for the PCIe write combine buffers.
	if(numThreads <= 0) {
		unsigned hw = std::thread::hardware_concurrency();
		numThreads = (int)(hw ? (hw < 4 ? hw : 4) : 2);
	}
	if(transferThreads.size() != (size_t)numThreads) {
		transferThreads.clear();
		for(int i = 0; i < numThreads; i++) {
			transferThreads.emplace_back(new MessageThread(PAR2ProcVulkan::transfer_slice));
			transferThreads.back()->name = "vulkan_transfer";
		}
		nextTransferThread.store(0, std::memory_order_relaxed);
	}
	if(!completionThread) {
		completionThread.reset(new MessageThread(PAR2ProcVulkan::completion_worker));
		completionThread->name = "vulkan_complete";
	}

	outputExponents.clear();

	// The batch size trades staging memory against arithmetic intensity: each
	// input read inside the kernel is applied to every output in the group, so
	// larger batches amortise the read further.
	inputBatchSize = inputGrouping ? inputGrouping : 16;
	setMinInputBatchSize(0);

	if(!gf || gfMethod != cksumMethod) {
		gf.reset(new Galois16Mul(cksumMethod));
		gfMethod = cksumMethod;
	}
	sliceSizeCksum = sliceSize + gf->info().cksumSize;

	reset_state();
	statBatchesStarted = 0;
	chooseGeometry();
	if(!buildPipeline()) return false;
	if(gpu_stats_enabled() && g_stats.wallStart == 0) g_stats.wallStart = now_s();
	return true;
}

void PAR2ProcVulkan::chooseGeometry() {
	// Shared memory holds one output group's tables across the whole input
	// batch, so the group width is bounded by what fits in it. Unlike the Metal
	// kernel, the shared array here is sized by the same constants, so lowering
	// the group width genuinely lowers the allocation.
	const size_t shared = impl->props.limits.maxComputeSharedMemorySize;
	const size_t perOutput = (size_t)inputBatchSize * GF16_LUT_ENTRIES
		* sizeof(gf16_lut_entry);

	unsigned opg = GF16_MAX_OUTPUTS_PER_GROUP;
	while(opg > 1 && perOutput * opg > shared)
		opg--;
	outputsPerGroup = opg;

	unsigned maxThreads = impl->props.limits.maxComputeWorkGroupInvocations;
	if(maxThreads > impl->props.limits.maxComputeWorkGroupSize[0])
		maxThreads = impl->props.limits.maxComputeWorkGroupSize[0];
	threadsPerGroup = std::min<unsigned>(maxThreads, 256);
	if(threadsPerGroup == 0) threadsPerGroup = 32;
}

bool PAR2ProcVulkan::buildPipeline() {
	// The geometry is baked into the pipeline through specialization constants,
	// so it has to be rebuilt whenever chooseGeometry() changes its mind.
	if(impl->pipeline) {
		impl->dev.vkDeviceWaitIdle(impl->device);
		impl->dev.vkDestroyPipeline(impl->device, impl->pipeline, NULL);
		impl->pipeline = VK_NULL_HANDLE;
	}

	const uint32_t specData[3] = {
		threadsPerGroup, (uint32_t)inputBatchSize, outputsPerGroup
	};
	VkSpecializationMapEntry entries[3];
	for(uint32_t i = 0; i < 3; i++) {
		entries[i].constantID = i;
		entries[i].offset = i * sizeof(uint32_t);
		entries[i].size = sizeof(uint32_t);
	}
	VkSpecializationInfo spec = {};
	spec.mapEntryCount = 3;
	spec.pMapEntries = entries;
	spec.dataSize = sizeof(specData);
	spec.pData = specData;

	VkComputePipelineCreateInfo ci = {};
	ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	ci.stage.module = impl->shader;
	ci.stage.pName = "main";
	ci.stage.pSpecializationInfo = &spec;
	ci.layout = impl->pipeLayout;

	return impl->dev.vkCreateComputePipelines(impl->device, VK_NULL_HANDLE, 1, &ci,
	                                          NULL, &impl->pipeline) == VK_SUCCESS;
}

void PAR2ProcVulkan::reset_state() {
	currentStagingArea = 0;
	currentStagingInputs = 0;
	stagingActiveCount = 0;
	for(auto& area : staging) {
		area.pending.store(0, std::memory_order_relaxed);
		area.submitCount.store(0, std::memory_order_relaxed);
		area.setIsActive(false);
	}
	processingAdd = false;
}

// ---- sizing --------------------------------------------------------------

void PAR2ProcVulkan::setSliceSize(size_t size) {
	sliceSize = size;
	if(gf) sliceSizeCksum = size + gf->info().cksumSize;
}

bool PAR2ProcVulkan::setCurrentSliceSize(size_t size) {
	setSliceSize(size);
	sliceSizeAligned = (sliceSizeCksum + GF16_VECTOR_BYTES - 1)
		& ~(size_t)(GF16_VECTOR_BYTES - 1);

	if(sliceSizeAligned > allocatedSliceSize)
		return reallocBuffers();
	return true;
}

bool PAR2ProcVulkan::setRecoverySlices(unsigned numSlices, const uint16_t* exponents) {
	outputExponents.clear();
	if(numSlices == 0) return true;

	// Default to 1, matching PAR2ProcCPU: an exponent of 0 would make every
	// coefficient 1, and also trips the output==0 add shortcut when the caller
	// intends to supply explicit coefficients per input instead.
	outputExponents.resize(numSlices, 1);
	if(exponents)
		std::memcpy(outputExponents.data(), exponents, numSlices * sizeof(uint16_t));

	return reallocBuffers();
}

bool PAR2ProcVulkan::reallocBuffers() {
	if(!initSuccess) return false;
	if(sliceSizeAligned == 0 || outputExponents.empty()) return true;

	impl->dev.vkDeviceWaitIdle(impl->device);

	const size_t numOutputs = outputExponents.size();
	const size_t outBytes = sliceSizeAligned * numOutputs;
	const size_t inBytes = sliceSizeAligned * inputBatchSize;
	const size_t lutBytes = numOutputs * inputBatchSize * GF16_LUT_ENTRIES
		* sizeof(gf16_lut_entry);

	// A storage buffer descriptor cannot address more than this, however much
	// was allocated.
	const uint64_t maxRange = impl->props.limits.maxStorageBufferRange;
	if(outBytes > maxRange || inBytes > maxRange || lutBytes > maxRange)
		return false; // caller falls back to the CPU backend

	// Unlike unified memory, VRAM is finite and overcommitting it does not
	// fail cleanly -- the driver starts paging over PCIe and throughput
	// collapses to something far worse than the CPU path. Refuse instead, and
	// let the caller fall back.
	const uint64_t budget = impl->deviceLocalMemory();
	if(budget) {
		const uint64_t wanted = (uint64_t)outBytes
			+ (uint64_t)(inBytes + lutBytes) * impl->areas.size();
		// Leave headroom: the display, other processes and the driver's own
		// allocations all come out of the same heap.
		if(wanted > budget - budget / 5) return false;
	}

	if(!impl->createBuffer(outBytes,
			VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, impl->devOutput, false))
		return false;

	// One descriptor set per staging area, so a dispatch does not have to wait
	// for the previous one's descriptors to be free.
	if(impl->descPool) {
		impl->dev.vkDestroyDescriptorPool(impl->device, impl->descPool, NULL);
		impl->descPool = VK_NULL_HANDLE;
	}
	VkDescriptorPoolSize poolSize = {};
	poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	poolSize.descriptorCount = (uint32_t)(3 * impl->areas.size());
	VkDescriptorPoolCreateInfo dpi = {};
	dpi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	dpi.maxSets = (uint32_t)impl->areas.size();
	dpi.poolSizeCount = 1;
	dpi.pPoolSizes = &poolSize;
	if(impl->dev.vkCreateDescriptorPool(impl->device, &dpi, NULL,
	                                    &impl->descPool) != VK_SUCCESS) return false;

	for(size_t i = 0; i < impl->areas.size(); i++) {
		VulkanArea& a = impl->areas[i];

		if(!impl->createBuffer(inBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
				a.hostInput, true))
			return false;
		if(!impl->createBuffer(inBytes,
				VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, a.devInput, false))
			return false;
		if(!impl->createBuffer(lutBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
				a.hostLut, true))
			return false;
		if(!impl->createBuffer(lutBytes,
				VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, a.devLut, false))
			return false;

		VkDescriptorSetAllocateInfo dsi = {};
		dsi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		dsi.descriptorPool = impl->descPool;
		dsi.descriptorSetCount = 1;
		dsi.pSetLayouts = &impl->setLayout;
		if(impl->dev.vkAllocateDescriptorSets(impl->device, &dsi, &a.set) != VK_SUCCESS)
			return false;

		VkDescriptorBufferInfo bufs[3] = {};
		bufs[0].buffer = impl->devOutput.buffer; bufs[0].range = VK_WHOLE_SIZE;
		bufs[1].buffer = a.devInput.buffer;      bufs[1].range = VK_WHOLE_SIZE;
		bufs[2].buffer = a.devLut.buffer;        bufs[2].range = VK_WHOLE_SIZE;
		VkWriteDescriptorSet writes[3] = {};
		for(uint32_t b = 0; b < 3; b++) {
			writes[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			writes[b].dstSet = a.set;
			writes[b].dstBinding = b;
			writes[b].descriptorCount = 1;
			writes[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			writes[b].pBufferInfo = &bufs[b];
		}
		impl->dev.vkUpdateDescriptorSets(impl->device, 3, writes, 0, NULL);

		if(a.cmd == VK_NULL_HANDLE) {
			VkCommandBufferAllocateInfo cbi = {};
			cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
			cbi.commandPool = impl->cmdPool;
			cbi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
			cbi.commandBufferCount = 1;
			if(impl->dev.vkAllocateCommandBuffers(impl->device, &cbi, &a.cmd) != VK_SUCCESS)
				return false;
		}
		if(a.fence == VK_NULL_HANDLE) {
			VkFenceCreateInfo fi = {};
			fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
			if(impl->dev.vkCreateFence(impl->device, &fi, NULL, &a.fence) != VK_SUCCESS)
				return false;
		}
		if(a.queries == VK_NULL_HANDLE && impl->haveTimestamps && gpu_stats_enabled()) {
			VkQueryPoolCreateInfo qpi = {};
			qpi.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
			qpi.queryType = VK_QUERY_TYPE_TIMESTAMP;
			qpi.queryCount = 2;
			if(impl->dev.vkCreateQueryPool(impl->device, &qpi, NULL, &a.queries) != VK_SUCCESS)
				a.queries = VK_NULL_HANDLE; // stats are optional; carry on
		}

		staging[i].procCoeffs.resize(numOutputs * inputBatchSize);
	}

	// One readback staging buffer, command pool and fence per transfer thread.
	if(impl->readback.size() != transferThreads.size()) {
		for(size_t i = 0; i < impl->readback.size(); i++) {
			impl->destroyBuffer(impl->readback[i].host);
			if(impl->readback[i].fence)
				impl->dev.vkDestroyFence(impl->device, impl->readback[i].fence, NULL);
			if(impl->readback[i].pool)
				impl->dev.vkDestroyCommandPool(impl->device, impl->readback[i].pool, NULL);
		}
		impl->readback.assign(transferThreads.size(), VulkanReadback());
	}
	for(size_t i = 0; i < impl->readback.size(); i++) {
		VulkanReadback& r = impl->readback[i];
		// HOST_CACHED as well, since this buffer is read back by the CPU and
		// uncached reads over PCIe are catastrophically slow. It is only a
		// preference: the fallback in createBuffer covers devices without it.
		if(!impl->createBuffer(sliceSizeAligned, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
				| VK_MEMORY_PROPERTY_HOST_CACHED_BIT, r.host, true)) {
			if(!impl->createBuffer(sliceSizeAligned, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
					VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
					r.host, true))
				return false;
		}
		if(r.pool == VK_NULL_HANDLE) {
			VkCommandPoolCreateInfo cpi = {};
			cpi.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
			cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
			cpi.queueFamilyIndex = impl->queueFamily;
			if(impl->dev.vkCreateCommandPool(impl->device, &cpi, NULL, &r.pool) != VK_SUCCESS)
				return false;
			VkCommandBufferAllocateInfo cbi = {};
			cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
			cbi.commandPool = r.pool;
			cbi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
			cbi.commandBufferCount = 1;
			if(impl->dev.vkAllocateCommandBuffers(impl->device, &cbi, &r.cmd) != VK_SUCCESS)
				return false;
		}
		if(r.fence == VK_NULL_HANDLE) {
			VkFenceCreateInfo fi = {};
			fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
			if(impl->dev.vkCreateFence(impl->device, &fi, NULL, &r.fence) != VK_SUCCESS)
				return false;
		}
	}

	allocatedSliceSize = sliceSizeAligned;
	reset_state();
	return true;
}

// ---- coefficients --------------------------------------------------------

void PAR2ProcVulkan::set_coeffs(PAR2ProcVulkanStaging& area, unsigned idx,
                                uint16_t inputNum) {
	uint16_t inputLog = gfmat_input_log(inputNum);
	for(unsigned o = 0; o < outputExponents.size(); o++)
		area.procCoeffs[idx + o * inputBatchSize] =
			gfmat_coeff_from_log(inputLog, outputExponents[o]);
}

void PAR2ProcVulkan::set_coeffs(PAR2ProcVulkanStaging& area, unsigned idx,
                                const uint16_t* inputCoeffs) {
	for(unsigned o = 0; o < outputExponents.size(); o++)
		area.procCoeffs[idx + o * inputBatchSize] = inputCoeffs[o];
}

// ---- transfer worker -----------------------------------------------------

namespace {
enum VulkanReqKind {
	VK_REQ_STAGE,    // copy an input slice into the host-visible buffer
	VK_REQ_READBACK, // copy an output slice back, verifying its checksum
	VK_REQ_BARRIER   // no work; completing it means the queue has drained
};

struct VulkanTransferReq {
	VulkanReqKind kind;

	PAR2ProcVulkan* parent;
	PAR2ProcVulkanImpl* impl;
	Galois16Mul* gf;

	void* local;
	size_t sliceLen;
	size_t totalLen;

	unsigned threadIdx; // which transfer thread will handle this

	// staging
	unsigned area;
	unsigned slot;
	size_t srcLen;
	std::promise<void> promPrep;

	// readback
	unsigned outputIndex;
	size_t outputOffset;
	std::promise<bool> promOut;
};

struct VulkanCompletionReq {
	PAR2ProcVulkan* parent;
	PAR2ProcVulkanImpl* impl;
	unsigned area;
	bool barrier;
	std::promise<bool> promDone;
};
} // namespace

unsigned PAR2ProcVulkan::pickTransferThread() {
	return nextTransferThread.fetch_add(1, std::memory_order_relaxed)
		% (unsigned)transferThreads.size();
}

void PAR2ProcVulkan::sendToTransfer(unsigned threadIdx, void* req) {
	transferThreads[threadIdx]->send(req);
}

void PAR2ProcVulkan::transfer_slice(ThreadMessageQueue<void*>& q) {
	VulkanTransferReq* req;
	while((req = static_cast<VulkanTransferReq*>(q.pop())) != NULL) {
		if(req->kind == VK_REQ_BARRIER) {
			req->promOut.set_value(true);
		} else if(req->kind == VK_REQ_READBACK) {
			PAR2ProcVulkanImpl* impl = req->impl;
			VulkanReadback& r = impl->readback[req->threadIdx];

			// Pull this slice out of VRAM into host-visible memory first;
			// unlike Metal there is no pointer to read directly.
			const uint64_t t0 = gpu_stats_enabled() ? now_ns() : 0;
			bool ok = false;

			VkCommandBufferBeginInfo bi = {};
			bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
			if(impl->dev.vkBeginCommandBuffer(r.cmd, &bi) == VK_SUCCESS) {
				// The compute writes are already fence-complete by the time
				// par2 asks for output, but make the dependency explicit
				// rather than relying on that.
				VkMemoryBarrier mb = {};
				mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
				mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
				mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
				impl->dev.vkCmdPipelineBarrier(r.cmd,
					VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
					VK_PIPELINE_STAGE_TRANSFER_BIT,
					0, 1, &mb, 0, NULL, 0, NULL);

				VkBufferCopy copy = {};
				copy.srcOffset = req->outputOffset;
				copy.dstOffset = 0;
				copy.size = req->totalLen;
				impl->dev.vkCmdCopyBuffer(r.cmd, impl->devOutput.buffer,
				                          r.host.buffer, 1, &copy);

				if(impl->dev.vkEndCommandBuffer(r.cmd) == VK_SUCCESS) {
					VkSubmitInfo si = {};
					si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
					si.commandBufferCount = 1;
					si.pCommandBuffers = &r.cmd;
					VkResult sr;
					{
						std::lock_guard<std::mutex> lock(impl->queueMutex);
						impl->dev.vkResetFences(impl->device, 1, &r.fence);
						sr = impl->dev.vkQueueSubmit(impl->queue, 1, &si, r.fence);
					}
					if(sr == VK_SUCCESS)
						ok = impl->dev.vkWaitForFences(impl->device, 1, &r.fence,
						                               VK_TRUE, UINT64_MAX) == VK_SUCCESS;
				}
			}

			if(ok) {
				// Verifies the GF16 checksum the kernel carried through the
				// multiply-add; a false here means the GPU produced bad data.
				ok = req->gf->copy_cksum_check(req->local, r.host.mapped,
				                               req->sliceLen) != 0;
			}
			if(gpu_stats_enabled()) g_stats.readbackNs += now_ns() - t0;
			req->promOut.set_value(ok);
		} else {
			if(req->local) {
				uint8_t* dst = (uint8_t*)req->impl->areas[req->area].hostInput.mapped
					+ (size_t)req->slot * req->parent->getAllocSliceSize();
				const uint64_t t0 = gpu_stats_enabled() ? now_ns() : 0;
				req->gf->copy_cksum(dst, req->local, req->srcLen, req->sliceLen);
				if(gpu_stats_enabled()) {
					g_stats.stageNs += now_ns() - t0;
					g_stats.stageBytes += req->srcLen;
				}
			}
			// Let the caller reuse its buffer before we consider dispatching,
			// so a slow dispatch cannot stall the reader.
			req->promPrep.set_value();
			req->parent->releaseStageToken(req->area);
		}
		delete req;
	}
}

// Waits on each dispatched batch's fence and releases its staging area. Vulkan
// has no completion callback, so this stands in for Metal's addCompletedHandler.
void PAR2ProcVulkan::completion_worker(ThreadMessageQueue<void*>& q) {
	VulkanCompletionReq* req;
	while((req = static_cast<VulkanCompletionReq*>(q.pop())) != NULL) {
		if(req->barrier) {
			req->promDone.set_value(true);
			delete req;
			continue;
		}

		PAR2ProcVulkanImpl* impl = req->impl;
		VulkanArea& a = impl->areas[req->area];
		impl->dev.vkWaitForFences(impl->device, 1, &a.fence, VK_TRUE, UINT64_MAX);

		if(gpu_stats_enabled() && a.queries != VK_NULL_HANDLE) {
			uint64_t ts[2] = {0, 0};
			if(impl->dev.vkGetQueryPoolResults(impl->device, a.queries, 0, 2,
					sizeof(ts), ts, sizeof(uint64_t),
					VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) == VK_SUCCESS
			   && ts[1] > ts[0]) {
				// Real time on the GPU, as opposed to wall time, which
				// includes queueing and any host-side stall.
				g_stats.gpuNs += (uint64_t)((ts[1] - ts[0]) * impl->timestampPeriod);
			}
			g_stats.dispatches++;
		} else if(gpu_stats_enabled()) {
			g_stats.dispatches++;
		}

		impl->dev.vkResetFences(impl->device, 1, &a.fence);
		req->parent->_batchCompleted(req->area);
		delete req;
	}
}

void PAR2ProcVulkan::_batchCompleted(unsigned area) {
	stagingActiveCount_dec();
	_setAreaActive(area, false);
}

// Blocks until every queued transfer and completion has been handled. The
// queues are FIFO, so a completed barrier means everything sent before it is
// done. Needed because MessageThread::end() signals the worker without joining
// it, and _deinit() releases the buffers those requests reference.
void PAR2ProcVulkan::drainTransfers() {
	std::vector<std::future<bool>> barriers;
	barriers.reserve(transferThreads.size() + 1);
	for(auto& t : transferThreads) {
		if(t->empty()) continue;
		VulkanTransferReq* req = new VulkanTransferReq();
		req->kind = VK_REQ_BARRIER;
		req->parent = this;
		req->impl = impl.get();
		req->gf = gf.get();
		barriers.push_back(req->promOut.get_future());
		t->send(req);
	}
	if(completionThread && !completionThread->empty()) {
		VulkanCompletionReq* req = new VulkanCompletionReq();
		req->parent = this;
		req->impl = impl.get();
		req->area = 0;
		req->barrier = true;
		barriers.push_back(req->promDone.get_future());
		completionThread->send(req);
	}
	for(auto& f : barriers) f.get();
}

// ---- batch completion ----------------------------------------------------

void PAR2ProcVulkan::beginBatch(PAR2ProcVulkanStaging& area) {
	area.submitCount.store(0, std::memory_order_relaxed);
	// The "batch open" token; released when the batch is closed.
	area.pending.store(1, std::memory_order_relaxed);
}

// Drops one reference to a batch. The last one out dispatches it, which may be
// a staging worker or the thread that closed the batch - whichever finishes
// last. submitCount is published with release and read with acquire so the
// dispatcher always sees the closed batch's size.
void PAR2ProcVulkan::releaseStageToken(unsigned areaIdx) {
	auto& area = staging[areaIdx];
	if(area.pending.fetch_sub(1, std::memory_order_acq_rel) != 1)
		return;
	unsigned n = area.submitCount.load(std::memory_order_acquire);
	if(n) run_kernel(areaIdx, n);
}

// ---- dispatch ------------------------------------------------------------

void PAR2ProcVulkan::run_kernel(unsigned area, unsigned numInputs) {
	std::lock_guard<std::mutex> dispatchLock(dispatchMutex);

	const unsigned numOutputs = (unsigned)outputExponents.size();
	auto& st = staging[area];
	VulkanArea& a = impl->areas[area];

	const uint64_t tEncode0 = gpu_stats_enabled() ? now_ns() : 0;

	// Build this batch's lookup tables, laid out [output][input][64] to match
	// the kernel's shared-memory staging.
	const uint64_t tLut0 = gpu_stats_enabled() ? now_ns() : 0;
	gf16_lut_entry* luts = (gf16_lut_entry*)a.hostLut.mapped;
	for(unsigned o = 0; o < numOutputs; o++) {
		for(unsigned i = 0; i < numInputs; i++) {
			gf16_build_lut(luts + ((size_t)o * numInputs + i) * GF16_LUT_ENTRIES,
			               st.procCoeffs[i + o * inputBatchSize]);
		}
	}

	// Whether this dispatch accumulates or overwrites is decided here, from the
	// flag as it stood before this batch - matching PAR2ProcCPU. Tracking it
	// separately would miss discardOutput(), which resets processingAdd between
	// chunks and must force the next dispatch to overwrite.
	const bool accumulate = processingAdd;
	processingAdd = true;

	if(gpu_stats_enabled()) g_stats.lutNs += now_ns() - tLut0;

	const uint32_t vecPerSlice = (uint32_t)(sliceSizeAligned / GF16_VECTOR_BYTES);
	GF16Params params = {
		numInputs, numOutputs, outputsPerGroup,
		vecPerSlice, vecPerSlice,
		accumulate ? 1u : 0u
	};

	const VkDeviceSize inBytes = (VkDeviceSize)numInputs * sliceSizeAligned;
	const VkDeviceSize lutBytes = (VkDeviceSize)numOutputs * numInputs
		* GF16_LUT_ENTRIES * sizeof(gf16_lut_entry);

	// The caller has already marked this area active and counted it in flight.
	// Every early return below must undo that, or waitForAdd() blocks forever
	// waiting for an area that will never be dispatched.
	VkCommandBufferBeginInfo bi = {};
	bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	if(impl->dev.vkBeginCommandBuffer(a.cmd, &bi) != VK_SUCCESS) {
		_batchCompleted(area);
		return;
	}

	const bool timing = gpu_stats_enabled() && a.queries != VK_NULL_HANDLE;
	if(timing) {
		impl->dev.vkCmdResetQueryPool(a.cmd, a.queries, 0, 2);
		impl->dev.vkCmdWriteTimestamp(a.cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
		                              a.queries, 0);
	}

	// Stage this batch across PCIe into device-local memory. The kernel must
	// read device memory; pointing it at the host buffer would trade the whole
	// arithmetic-intensity advantage for PCIe latency per access.
	VkBufferCopy inCopy = {};
	inCopy.size = inBytes;
	impl->dev.vkCmdCopyBuffer(a.cmd, a.hostInput.buffer, a.devInput.buffer, 1, &inCopy);
	VkBufferCopy lutCopy = {};
	lutCopy.size = lutBytes;
	impl->dev.vkCmdCopyBuffer(a.cmd, a.hostLut.buffer, a.devLut.buffer, 1, &lutCopy);

	// Two dependencies in one barrier: the copies above must land before the
	// kernel reads them, and the previous batch's accumulation into the output
	// must land before this one reads-modifies-writes it. The first scope of a
	// barrier covers everything previously submitted to this queue, which is
	// what makes successive batches accumulate rather than race.
	VkMemoryBarrier mb = {};
	mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
	mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT;
	mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
	impl->dev.vkCmdPipelineBarrier(a.cmd,
		VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		0, 1, &mb, 0, NULL, 0, NULL);

	impl->dev.vkCmdBindPipeline(a.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, impl->pipeline);
	impl->dev.vkCmdBindDescriptorSets(a.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
	                                  impl->pipeLayout, 0, 1, &a.set, 0, NULL);
	impl->dev.vkCmdPushConstants(a.cmd, impl->pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT,
	                             0, sizeof(params), &params);

	uint32_t groupsX = (vecPerSlice + threadsPerGroup - 1) / threadsPerGroup;
	if(groupsX == 0) groupsX = 1;
	if(groupsX > 256) groupsX = 256; // grid-stride loop covers the rest
	const uint32_t groupsY = (numOutputs + outputsPerGroup - 1) / outputsPerGroup;
	impl->dev.vkCmdDispatch(a.cmd, groupsX, groupsY, 1);

	if(timing)
		impl->dev.vkCmdWriteTimestamp(a.cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
		                              a.queries, 1);

	if(impl->dev.vkEndCommandBuffer(a.cmd) != VK_SUCCESS) {
		_batchCompleted(area);
		return;
	}

	VkSubmitInfo si = {};
	si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	si.commandBufferCount = 1;
	si.pCommandBuffers = &a.cmd;
	{
		std::lock_guard<std::mutex> lock(impl->queueMutex);
		if(impl->dev.vkQueueSubmit(impl->queue, 1, &si, a.fence) != VK_SUCCESS) {
			_batchCompleted(area);
			return;
		}
	}

	VulkanCompletionReq* creq = new VulkanCompletionReq();
	creq->parent = this;
	creq->impl = impl.get();
	creq->area = area;
	creq->barrier = false;
	completionThread->send(creq);

	if(gpu_stats_enabled()) g_stats.encodeNs += now_ns() - tEncode0;
}

// ---- input ---------------------------------------------------------------

PAR2ProcBackendAddResult PAR2ProcVulkan::canAdd() const {
	if(staging[currentStagingArea].getIsActive()) return PROC_ADD_FULL;
	return stagingActiveCount_get() < staging.size() - 1 ? PROC_ADD_OK : PROC_ADD_OK_BUSY;
}

void PAR2ProcVulkan::waitForAdd() {
	IPAR2ProcBackend::_waitForAdd(staging[currentStagingArea]);
}

template<typename T>
FUTURE_RETURN_T PAR2ProcVulkan::_addInput(const void* buffer, size_t size,
                                          T inputNumOrCoeffs, bool flush) {
	auto& area = staging[currentStagingArea];
	assert(!area.getIsActive());

	if(currentStagingInputs == 0) beginBatch(area);
	set_coeffs(area, currentStagingInputs, inputNumOrCoeffs);

	VulkanTransferReq* req = new VulkanTransferReq();
	req->kind = VK_REQ_STAGE;
	req->parent = this;
	req->impl = impl.get();
	req->gf = gf.get();
	req->local = (void*)buffer;
	req->srcLen = size;
	req->sliceLen = sliceSize;
	req->totalLen = sliceSizeCksum;
	req->area = currentStagingArea;
	req->slot = currentStagingInputs;

	currentStagingInputs++;
	const unsigned submit = (flush || currentStagingInputs == inputBatchSize || (
		// nothing in flight: submit early rather than leave the GPU idle
		stagingActiveCount_get() == 0 && staging.size() > 1
			&& currentStagingInputs >= minInBatchSize
	)) ? currentStagingInputs : 0;

	// This slice's reference, taken before it is queued so a worker cannot
	// complete it and close the batch prematurely.
	area.pending.fetch_add(1, std::memory_order_relaxed);

	const unsigned areaIdx = currentStagingArea;
	if(submit) {
		stagingActiveCount_inc();
		area.setIsActive(true);
		statBatchesStarted++;
		currentStagingInputs = 0;
		if(++currentStagingArea == staging.size())
			currentStagingArea = 0;
	}

	req->threadIdx = pickTransferThread();
	auto future = req->promPrep.get_future();
	sendToTransfer(req->threadIdx, req);

	if(submit) {
		area.submitCount.store(submit, std::memory_order_release);
		releaseStageToken(areaIdx); // drops the batch-open token
	}
	return future;
}

FUTURE_RETURN_T PAR2ProcVulkan::addInput(const void* buffer, size_t size,
                                         uint16_t inputNum, bool flush) {
	return _addInput(buffer, size, inputNum, flush);
}

FUTURE_RETURN_T PAR2ProcVulkan::addInput(const void* buffer, size_t size,
                                         const uint16_t* coeffs, bool flush) {
	return _addInput(buffer, size, coeffs, flush);
}

void PAR2ProcVulkan::dummyInput(uint16_t inputNum, bool flush) {
	// Benchmarking hook: account for an input without transferring it.
	auto& area = staging[currentStagingArea];
	if(currentStagingInputs == 0) beginBatch(area);
	set_coeffs(area, currentStagingInputs, inputNum);
	currentStagingInputs++;
	const unsigned submit = (flush || currentStagingInputs == inputBatchSize)
		? currentStagingInputs : 0;
	if(submit) {
		stagingActiveCount_inc();
		area.setIsActive(true);
		statBatchesStarted++;
		currentStagingInputs = 0;
		const unsigned areaIdx = currentStagingArea;
		if(++currentStagingArea == staging.size()) currentStagingArea = 0;
		area.submitCount.store(submit, std::memory_order_release);
		releaseStageToken(areaIdx);
	}
}

bool PAR2ProcVulkan::fillInput(const void* buffer) {
	(void)buffer;
	return false; // benchmarking hook, not needed by par2cmdline
}

void PAR2ProcVulkan::flush() {
	if(!currentStagingInputs) return;

	const unsigned areaIdx = currentStagingArea;
	auto& area = staging[areaIdx];
	const unsigned submit = currentStagingInputs;

	stagingActiveCount_inc();
	area.setIsActive(true);
	statBatchesStarted++;
	currentStagingInputs = 0;
	if(++currentStagingArea == staging.size())
		currentStagingArea = 0;

	// No slice to stage; releasing the batch-open token dispatches once the
	// already-queued slices have finished.
	area.submitCount.store(submit, std::memory_order_release);
	releaseStageToken(areaIdx);
}

// ---- output --------------------------------------------------------------

FUTURE_RETURN_BOOL_T PAR2ProcVulkan::getOutput(unsigned index, void* output) {
	// Readback resources are sized in reallocBuffers(), which is a no-op until
	// there is both a slice size and an output count. Asking for output before
	// then is a caller error, but it must not index an empty vector.
	if(impl->readback.empty() || impl->devOutput.buffer == VK_NULL_HANDLE) {
		std::promise<bool> p;
		p.set_value(false);
		return p.get_future();
	}

	VulkanTransferReq* req = new VulkanTransferReq();
	req->kind = VK_REQ_READBACK;
	req->parent = this;
	req->impl = impl.get();
	req->gf = gf.get();
	req->local = output;
	req->sliceLen = sliceSize;
	req->totalLen = sliceSizeAligned;
	req->outputIndex = index;
	req->outputOffset = (size_t)index * sliceSizeAligned;

	req->threadIdx = pickTransferThread();
	auto future = req->promOut.get_future();
	sendToTransfer(req->threadIdx, req);
	return future;
}
