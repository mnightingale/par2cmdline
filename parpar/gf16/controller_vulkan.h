#ifndef __GF16_CONTROLLER_VULKAN
#define __GF16_CONTROLLER_VULKAN

// Vulkan compute backend for GF16 multiply-add.
//
// Mirrors PAR2ProcMetal: staging areas accumulate a batch of input slices, and
// once a batch is full it is dispatched while the next batch is staged. Slices
// carry ParPar's GF16 checksum, which the kernel propagates through the
// multiply-add and getOutput() verifies, so a compute or memory fault on the
// GPU is detected rather than silently producing a corrupt repair.
//
// Where this differs from the Metal backend, it is because the memory is not
// shared. Metal staged into a buffer the GPU could read directly; here every
// slice crosses PCIe:
//
//   host-visible staging buffer  --vkCmdCopyBuffer-->  device-local buffer
//                                                              |
//                                                        kernel reads this
//
// The kernel must read device-local memory, not host memory. Reading host
// memory over PCIe per access would make the whole exercise pointless, since
// the arithmetic intensity is what the design is buying.
//
// Vulkan handles are kept behind a pimpl so this header stays pure C++ and can
// be included without the Vulkan headers on the include path.

#include "controller.h"
#include "threadqueue.h"
#include "gf16mul.h"
#include <memory>
#include <atomic>
#include <mutex>
#include <vector>

#ifdef USE_LIBUV
// As with the Metal backend: only the std::future path is implemented, and an
// untested concurrency path is worse than none.
# error "PAR2ProcVulkan implements only the std::future path; USE_LIBUV is unsupported."
#endif

class PAR2ProcVulkanStaging : public IPAR2ProcStaging {
public:
	// Staging is spread over several worker threads, so the batch is dispatched
	// by whichever worker finishes last. `pending` counts outstanding slices
	// plus one "batch open" token, held until the batch is closed so an
	// early-finishing slice cannot dispatch before the rest are queued.
	std::atomic<unsigned> pending;
	std::atomic<unsigned> submitCount; // inputs to dispatch, 0 while open

	PAR2ProcVulkanStaging() : IPAR2ProcStaging(), pending(0), submitCount(0) {}
};

// Holds the Vulkan objects; defined in controller_vulkan.cpp.
struct PAR2ProcVulkanImpl;

class PAR2ProcVulkan : public IPAR2ProcBackend {
private:
	std::unique_ptr<PAR2ProcVulkanImpl> impl;

	bool initSuccess;
	int deviceId;

	size_t sliceSize;        // live bytes per slice
	size_t sliceSizeCksum;   // sliceSize + checksum
	size_t sliceSizeAligned; // padded to the kernel's vector width
	size_t allocatedSliceSize;

	unsigned outputsPerGroup; // outputs handled per workgroup
	unsigned threadsPerGroup;

	std::unique_ptr<Galois16Mul> gf;
	Galois16Methods gfMethod;

	std::vector<PAR2ProcVulkanStaging> staging;

	// Staging (copy + GF16 checksum into the host-visible buffer) is the
	// host-side bottleneck, so it runs across several threads rather than one.
	std::vector<std::unique_ptr<MessageThread>> transferThreads;
	std::atomic<unsigned> nextTransferThread;
	int numThreads;
	static void transfer_slice(ThreadMessageQueue<void*>& q);
	// Split, because a readback request has to carry the index of the thread
	// that will handle it -- that is how the worker finds its own readback
	// buffer -- and the field must be stamped on before the request is sent.
	unsigned pickTransferThread();
	void sendToTransfer(unsigned threadIdx, void* req);

	// Vulkan signals completion with a fence rather than a callback, so one
	// thread does nothing but wait on the fence of each dispatched batch and
	// release its staging area. Batches complete in submission order on a
	// single queue, so a FIFO of area indices is enough.
	std::unique_ptr<MessageThread> completionThread;
	static void completion_worker(ThreadMessageQueue<void*>& q);

	// Serialises dispatch: the accumulate-vs-overwrite decision and the queue
	// submission must stay in the same order, or a batch could XOR into the
	// output and then be discarded by a later overwrite.
	std::mutex dispatchMutex;

	void beginBatch(PAR2ProcVulkanStaging& area);
	void releaseStageToken(unsigned area);

	void set_coeffs(PAR2ProcVulkanStaging& area, unsigned idx, uint16_t inputNum);
	void set_coeffs(PAR2ProcVulkanStaging& area, unsigned idx, const uint16_t* coeffs);
	template<typename T> FUTURE_RETURN_T _addInput(const void* buffer, size_t size,
	                                               T inputNumOrCoeffs, bool flush);

	void run_kernel(unsigned area, unsigned numInputs) override;
	bool reallocBuffers();
	bool buildPipeline();
	void reset_state();
	void drainTransfers();
	void chooseGeometry();

	PAR2ProcVulkan(const PAR2ProcVulkan&);
	PAR2ProcVulkan& operator=(const PAR2ProcVulkan&);

public:
	explicit PAR2ProcVulkan(int deviceId = -1, int stagingAreas = 2);
	~PAR2ProcVulkan();

	// True when a device was found and the queue and shader module built.
	bool isAvailable() const { return initSuccess; }

	bool init(unsigned inputGrouping = 0, Galois16Methods cksumMethod = GF16_AUTO);

	// Number of staging threads; must be set before init(). 0 picks a default.
	void setNumThreads(int threads);
	int getNumThreads() const { return numThreads; }

	void setSliceSize(size_t size) override;
	bool setCurrentSliceSize(size_t size) override;
	bool setRecoverySlices(unsigned numSlices, const uint16_t* exponents = NULL) override;

	PAR2ProcBackendAddResult canAdd() const override;
	FUTURE_RETURN_T addInput(const void* buffer, size_t size, uint16_t inputNum,
	                         bool flush) override;
	FUTURE_RETURN_T addInput(const void* buffer, size_t size, const uint16_t* coeffs,
	                         bool flush) override;
	void dummyInput(uint16_t inputNum, bool flush = false) override;
	bool fillInput(const void* buffer) override;
	void flush() override;
	FUTURE_RETURN_BOOL_T getOutput(unsigned index, void* output) override;

	void waitForAdd() override;
	FUTURE_RETURN_T endInput() override {
		return IPAR2ProcBackend::_endInput(staging);
	}
	void processing_finished() override {}

	void _deinit() override;
	void freeProcessingMem() override;

	inline void _setAreaActive(int area, bool active) {
		staging[area].setIsActive(active);
	}

	const char* getMethodName() const;
	size_t getAllocSliceSize() const { return sliceSizeAligned; }
	unsigned getStagingAreas() const { return (unsigned)staging.size(); }
	unsigned getOutputGrouping() const { return outputsPerGroup; }

	// Marks a dispatched batch complete once its fence signals. Called from the
	// completion thread; public only because that thread is a free function.
	void _batchCompleted(unsigned area);
};

#endif // defined(__GF16_CONTROLLER_VULKAN)
