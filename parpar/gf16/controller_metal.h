#ifndef __GF16_CONTROLLER_METAL
#define __GF16_CONTROLLER_METAL

// Metal compute backend for GF16 multiply-add.
//
// Structurally mirrors PAR2ProcCPU/PAR2ProcOCL: staging areas accumulate a
// batch of input slices, and once a batch is full it is dispatched to the GPU
// while the next batch is staged. Slices carry ParPar's GF16 checksum, which
// the kernel propagates through the multiply-add and getOutput() verifies, so
// a compute or memory fault on the GPU is detected rather than silently
// producing a corrupt repair.
//
// Objective-C types are kept behind a pimpl so this header stays pure C++ and
// can be included from ordinary translation units.

#include "controller.h"
#include "threadqueue.h"
#include "gf16mul.h"
#include <memory>
#include <atomic>
#include <mutex>
#include <vector>

#ifdef USE_LIBUV
// ParPar upstream drives backends through libuv rather than std::future. That
// path is deliberately not implemented here: it cannot be exercised by this
// build, and shipping an untested concurrency path is worse than shipping
// none. Fail at compile time rather than at runtime.
# error "PAR2ProcMetal implements only the std::future path; USE_LIBUV is unsupported."
#endif

class PAR2ProcMetalStaging : public IPAR2ProcStaging {
public:
	// Staging is spread over several worker threads, so the batch is dispatched
	// by whichever worker finishes last rather than by a fixed one. `pending`
	// counts outstanding slices plus one "batch open" token, held until the
	// batch is closed so an early-finishing slice cannot trigger a dispatch
	// before the remaining slices have even been queued.
	std::atomic<unsigned> pending;
	std::atomic<unsigned> submitCount; // inputs to dispatch, 0 while open

	PAR2ProcMetalStaging() : IPAR2ProcStaging(), pending(0), submitCount(0) {}
};

// Holds the Objective-C objects; defined in controller_metal.mm.
struct PAR2ProcMetalImpl;

class PAR2ProcMetal : public IPAR2ProcBackend {
private:
	std::unique_ptr<PAR2ProcMetalImpl> impl;

	bool initSuccess;
	int deviceId;

	size_t sliceSize;        // live bytes per slice
	size_t sliceSizeCksum;   // sliceSize + checksum
	size_t sliceSizeAligned; // padded to the kernel's vector width
	size_t allocatedSliceSize;

	unsigned outputsPerGroup; // outputs handled per threadgroup
	unsigned threadsPerGroup;

	std::unique_ptr<Galois16Mul> gf;
	Galois16Methods gfMethod;

	std::vector<PAR2ProcMetalStaging> staging;

	// Staging (copy + GF16 checksum of each slice into device memory) is the
	// host-side bottleneck, so it runs across several threads rather than one.
	std::vector<std::unique_ptr<MessageThread>> transferThreads;
	std::atomic<unsigned> nextTransferThread;
	int numThreads;
	static void transfer_slice(ThreadMessageQueue<void*>& q);
	void sendToTransfer(void* req);

	// Serialises kernel dispatch: the accumulate-vs-overwrite decision and the
	// command buffer commit must stay in the same order, or a batch could XOR
	// into a buffer that a later overwrite then discards.
	std::mutex dispatchMutex;

	void beginBatch(PAR2ProcMetalStaging& area);
	void releaseStageToken(unsigned area);

	void set_coeffs(PAR2ProcMetalStaging& area, unsigned idx, uint16_t inputNum);
	void set_coeffs(PAR2ProcMetalStaging& area, unsigned idx, const uint16_t* coeffs);
	template<typename T> FUTURE_RETURN_T _addInput(const void* buffer, size_t size,
	                                               T inputNumOrCoeffs, bool flush);

	void run_kernel(unsigned area, unsigned numInputs) override;
	bool reallocBuffers();
	void reset_state();
	void drainTransfers();
	void chooseGeometry();

	PAR2ProcMetal(const PAR2ProcMetal&);
	PAR2ProcMetal& operator=(const PAR2ProcMetal&);

public:
	explicit PAR2ProcMetal(int deviceId = -1, int stagingAreas = 2);
	~PAR2ProcMetal();

	// True when a device was found and the kernel pipeline built.
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
};

#endif // defined(__GF16_CONTROLLER_METAL)
