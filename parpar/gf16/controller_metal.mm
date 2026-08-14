#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "controller_metal.h"
#include "gfmat_coeff.h"
#include <cassert>
#include <cstring>
#include <algorithm>

// Compiled from gf16_metal.metal by the build; defines gf16_metal_lib[].
#include "gf16_metal_lib.h"

// Must match the kernel's definitions.
#define GF16_LUT_ENTRIES 64
#define GF16_MAX_OUTPUTS_PER_GROUP 8
#define GF16_VECTOR_BYTES 16 // uint4

struct GF16Params {
	uint32_t numInputs;
	uint32_t numOutputs;
	uint32_t outputsPerGroup;
	uint32_t vecPerSlice;
	uint32_t sliceStrideVec;
	uint32_t accumulate;
};

struct PAR2ProcMetalImpl {
	id<MTLDevice> device;
	id<MTLCommandQueue> queue;
	id<MTLComputePipelineState> pipeline;
	std::vector<id<MTLBuffer>> stagingInput; // one per staging area
	std::vector<id<MTLBuffer>> stagingLut;   // one per staging area
	id<MTLBuffer> output;
	NSString* deviceName;

	PAR2ProcMetalImpl() : device(nil), queue(nil), pipeline(nil), output(nil), deviceName(nil) {}
};

// ---- GF(2^16) helpers for lookup-table construction ----------------------
//
// Only used to build the nibble tables, which is negligible next to the data
// volume, so clarity beats cleverness here.

static const uint32_t GF16_POLY = 0x1100b;

static inline uint16_t gf16_double(uint16_t v) {
	uint32_t x = (uint32_t)v << 1;
	if(x & 0x10000) x ^= GF16_POLY;
	return (uint16_t)x;
}

// Four 16-entry tables for coefficient c: L[j*16 + n] = c * (n << 4j).
//
// Multiplication is linear over GF(2) in its second operand, so entry n is the
// XOR of the entries for each set bit of n. That builds a table from four
// doublings plus eleven XORs rather than sixteen full multiplies.
static void gf16_build_lut(uint16_t* L, uint16_t c) {
	L[0] = 0;
	L[1] = c;
	L[2] = gf16_double(L[1]);
	L[4] = gf16_double(L[2]);
	L[8] = gf16_double(L[4]);
	for(unsigned n = 3; n < 16; n++) {
		if(n == 4 || n == 8) continue; // already a power of two
		L[n] = L[n & (n - 1)] ^ L[n & (unsigned)(-(int)n)];
	}
	// Each subsequent table is the previous one multiplied by x^4.
	for(unsigned j = 1; j < 4; j++) {
		uint16_t* prev = L + (j - 1) * 16;
		uint16_t* cur = L + j * 16;
		for(unsigned n = 0; n < 16; n++)
			cur[n] = gf16_double(gf16_double(gf16_double(gf16_double(prev[n]))));
	}
}

// ---- construction --------------------------------------------------------

PAR2ProcMetal::PAR2ProcMetal(int _deviceId, int stagingAreas)
: IPAR2ProcBackend(), impl(new PAR2ProcMetalImpl()), initSuccess(false),
  deviceId(_deviceId), sliceSize(0), sliceSizeCksum(0), sliceSizeAligned(0),
  allocatedSliceSize(0), outputsPerGroup(4), threadsPerGroup(256),
  gf(nullptr), gfMethod(GF16_AUTO), staging(stagingAreas), outputDirty(false),
  transferThread(PAR2ProcMetal::transfer_slice)
{
	transferThread.name = "metal_transfer";

	@autoreleasepool {
		NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
		if(!devices || [devices count] == 0) return;
		NSUInteger idx = (_deviceId < 0) ? 0 : (NSUInteger)_deviceId;
		if(idx >= [devices count]) return;

		impl->device = devices[idx];
		impl->deviceName = [[impl->device name] copy];
		impl->queue = [impl->device newCommandQueue];
		if(!impl->queue) return;

		NSError* err = nil;
		dispatch_data_t libData = dispatch_data_create(
			gf16_metal_lib, sizeof(gf16_metal_lib),
			dispatch_get_main_queue(), DISPATCH_DATA_DESTRUCTOR_DEFAULT);
		id<MTLLibrary> lib = [impl->device newLibraryWithData:libData error:&err];
		if(!lib) return;

		id<MTLFunction> fn = [lib newFunctionWithName:@"gf16_muladd"];
		if(!fn) return;
		impl->pipeline = [impl->device newComputePipelineStateWithFunction:fn error:&err];
		if(!impl->pipeline) return;

		impl->stagingInput.resize(staging.size(), nil);
		impl->stagingLut.resize(staging.size(), nil);
		initSuccess = true;
	}
}

PAR2ProcMetal::~PAR2ProcMetal() {
	deinit();
}

const char* PAR2ProcMetal::getMethodName() const {
	return "Metal (nibble LUT)";
}

void PAR2ProcMetal::_deinit() {
	@autoreleasepool {
		impl->stagingInput.assign(impl->stagingInput.size(), nil);
		impl->stagingLut.assign(impl->stagingLut.size(), nil);
		impl->output = nil;
	}
	allocatedSliceSize = 0;
}

void PAR2ProcMetal::freeProcessingMem() {
	@autoreleasepool {
		impl->output = nil;
	}
}

bool PAR2ProcMetal::init(unsigned inputGrouping, Galois16Methods cksumMethod) {
	if(!initSuccess) return false;

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
	return true;
}

void PAR2ProcMetal::chooseGeometry() {
	// Threadgroup memory holds the tables for one output group across the whole
	// input batch, so the group width is bounded by what fits in it.
	const NSUInteger tgMem = [impl->device maxThreadgroupMemoryLength];
	const size_t perOutput = (size_t)inputBatchSize * GF16_LUT_ENTRIES * sizeof(uint16_t);

	unsigned opg = GF16_MAX_OUTPUTS_PER_GROUP;
	while(opg > 1 && perOutput * opg > (size_t)tgMem)
		opg--;
	outputsPerGroup = opg;

	NSUInteger maxThreads = [impl->pipeline maxTotalThreadsPerThreadgroup];
	threadsPerGroup = (unsigned)std::min<NSUInteger>(maxThreads, 256);
	if(threadsPerGroup == 0) threadsPerGroup = 32;
}

void PAR2ProcMetal::reset_state() {
	currentStagingArea = 0;
	currentStagingInputs = 0;
	stagingActiveCount = 0;
	for(auto& area : staging) {
		area.stagedInputs = 0;
		area.setIsActive(false);
	}
	processingAdd = false;
	outputDirty = false;
}

// ---- sizing --------------------------------------------------------------

void PAR2ProcMetal::setSliceSize(size_t size) {
	sliceSize = size;
	if(gf) sliceSizeCksum = size + gf->info().cksumSize;
}

bool PAR2ProcMetal::setCurrentSliceSize(size_t size) {
	setSliceSize(size);
	sliceSizeAligned = (sliceSizeCksum + GF16_VECTOR_BYTES - 1)
		& ~(size_t)(GF16_VECTOR_BYTES - 1);

	if(sliceSizeAligned > allocatedSliceSize)
		return reallocBuffers();
	return true;
}

bool PAR2ProcMetal::setRecoverySlices(unsigned numSlices, const uint16_t* exponents) {
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

bool PAR2ProcMetal::reallocBuffers() {
	if(!initSuccess) return false;
	if(sliceSizeAligned == 0 || outputExponents.empty()) return true; // nothing to size yet

	@autoreleasepool {
		const NSUInteger maxAlloc = [impl->device maxBufferLength];
		const size_t outBytes = sliceSizeAligned * outputExponents.size();
		const size_t inBytes = sliceSizeAligned * inputBatchSize;
		const size_t lutBytes = (size_t)outputExponents.size() * inputBatchSize
			* GF16_LUT_ENTRIES * sizeof(uint16_t);

		if(outBytes > maxAlloc || inBytes > maxAlloc || lutBytes > maxAlloc)
			return false; // caller falls back to the CPU backend

		// Shared storage is zero-copy on unified memory, so staging costs a
		// memcpy rather than a PCIe transfer.
		const MTLResourceOptions opts = MTLResourceStorageModeShared;

		impl->output = [impl->device newBufferWithLength:outBytes options:opts];
		if(!impl->output) return false;

		for(size_t i = 0; i < staging.size(); i++) {
			impl->stagingInput[i] = [impl->device newBufferWithLength:inBytes options:opts];
			impl->stagingLut[i] = [impl->device newBufferWithLength:lutBytes options:opts];
			if(!impl->stagingInput[i] || !impl->stagingLut[i]) return false;
			staging[i].procCoeffs.resize((size_t)outputExponents.size() * inputBatchSize);
		}
		allocatedSliceSize = sliceSizeAligned;
	}
	reset_state();
	return true;
}

// ---- coefficients --------------------------------------------------------

void PAR2ProcMetal::set_coeffs(PAR2ProcMetalStaging& area, unsigned idx, uint16_t inputNum) {
	uint16_t inputLog = gfmat_input_log(inputNum);
	for(unsigned o = 0; o < outputExponents.size(); o++)
		area.procCoeffs[idx + o * inputBatchSize] =
			gfmat_coeff_from_log(inputLog, outputExponents[o]);
}

void PAR2ProcMetal::set_coeffs(PAR2ProcMetalStaging& area, unsigned idx,
                               const uint16_t* inputCoeffs) {
	for(unsigned o = 0; o < outputExponents.size(); o++)
		area.procCoeffs[idx + o * inputBatchSize] = inputCoeffs[o];
}

// ---- transfer worker -----------------------------------------------------

struct MetalTransferReq {
	bool finish; // false = stage an input, true = read back an output

	PAR2ProcMetal* parent;
	PAR2ProcMetalImpl* impl;
	Galois16Mul* gf;

	void* local;
	size_t sliceLen;
	size_t totalLen;

	// staging
	unsigned area;
	unsigned slot;
	size_t srcLen;
	unsigned submitInputs; // non-zero: dispatch the batch after staging
	std::promise<void> promPrep;

	// readback
	unsigned outputIndex;
	size_t outputOffset;
	std::promise<bool> promOut;
};

void PAR2ProcMetal::transfer_slice(ThreadMessageQueue<void*>& q) {
	MetalTransferReq* req;
	while((req = static_cast<MetalTransferReq*>(q.pop())) != NULL) {
		if(req->finish) {
			const uint8_t* src = (const uint8_t*)[req->impl->output contents]
				+ req->outputOffset;
			// Verifies the GF16 checksum the kernel carried through the
			// multiply-add; a false here means the GPU produced bad data.
			int ok = req->gf->copy_cksum_check(req->local, src, req->sliceLen);
			req->promOut.set_value(ok != 0);
		} else {
			if(req->local) {
				uint8_t* dst = (uint8_t*)[req->impl->stagingInput[req->area] contents]
					+ (size_t)req->slot * req->parent->getAllocSliceSize();
				req->gf->copy_cksum(dst, req->local, req->srcLen, req->sliceLen);
			}
			if(req->submitInputs)
				req->parent->run_kernel(req->area, req->submitInputs);
			req->promPrep.set_value();
		}
		delete req;
	}
}

// ---- dispatch ------------------------------------------------------------

void PAR2ProcMetal::run_kernel(unsigned area, unsigned numInputs) {
	@autoreleasepool {
		const unsigned numOutputs = (unsigned)outputExponents.size();
		auto& st = staging[area];

		// Build this batch's lookup tables, laid out [output][input][64] to
		// match the kernel's threadgroup staging.
		uint16_t* luts = (uint16_t*)[impl->stagingLut[area] contents];
		for(unsigned o = 0; o < numOutputs; o++) {
			for(unsigned i = 0; i < numInputs; i++) {
				gf16_build_lut(luts + ((size_t)o * numInputs + i) * GF16_LUT_ENTRIES,
				               st.procCoeffs[i + o * inputBatchSize]);
			}
		}

		const uint32_t vecPerSlice = (uint32_t)(sliceSizeAligned / GF16_VECTOR_BYTES);
		GF16Params params = {
			numInputs, numOutputs, outputsPerGroup,
			vecPerSlice, vecPerSlice,
			outputDirty ? 1u : 0u
		};

		id<MTLCommandBuffer> cb = [impl->queue commandBuffer];
		id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
		[enc setComputePipelineState:impl->pipeline];
		[enc setBuffer:impl->output offset:0 atIndex:0];
		[enc setBuffer:impl->stagingInput[area] offset:0 atIndex:1];
		[enc setBuffer:impl->stagingLut[area] offset:0 atIndex:2];
		[enc setBytes:&params length:sizeof(params) atIndex:3];
		[enc setThreadgroupMemoryLength:
			(NSUInteger)outputsPerGroup * numInputs * GF16_LUT_ENTRIES * sizeof(uint16_t)
			atIndex:0];

		NSUInteger groupsX = (vecPerSlice + threadsPerGroup - 1) / threadsPerGroup;
		if(groupsX == 0) groupsX = 1;
		if(groupsX > 256) groupsX = 256; // grid-stride loop covers the rest
		NSUInteger groupsY = (numOutputs + outputsPerGroup - 1) / outputsPerGroup;

		[enc dispatchThreadgroups:MTLSizeMake(groupsX, groupsY, 1)
			threadsPerThreadgroup:MTLSizeMake(threadsPerGroup, 1, 1)];
		[enc endEncoding];

		// Command buffers on one queue execute in commit order, so batches
		// accumulate into the output buffer without racing each other.
		PAR2ProcMetal* self = this;
		[cb addCompletedHandler:^(id<MTLCommandBuffer> buf) {
			(void)buf;
			self->stagingActiveCount_dec();
			self->_setAreaActive(area, false);
		}];
		[cb commit];

		outputDirty = true;
	}
}

// ---- input ---------------------------------------------------------------

PAR2ProcBackendAddResult PAR2ProcMetal::canAdd() const {
	if(staging[currentStagingArea].getIsActive()) return PROC_ADD_FULL;
	return stagingActiveCount_get() < staging.size() - 1 ? PROC_ADD_OK : PROC_ADD_OK_BUSY;
}

void PAR2ProcMetal::waitForAdd() {
	IPAR2ProcBackend::_waitForAdd(staging[currentStagingArea]);
}

template<typename T>
FUTURE_RETURN_T PAR2ProcMetal::_addInput(const void* buffer, size_t size,
                                         T inputNumOrCoeffs, bool flush) {
	auto& area = staging[currentStagingArea];
	assert(!area.getIsActive());

	set_coeffs(area, currentStagingInputs, inputNumOrCoeffs);

	MetalTransferReq* req = new MetalTransferReq();
	req->finish = false;
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
	req->submitInputs = (flush || currentStagingInputs == inputBatchSize || (
		// nothing in flight: submit early rather than leave the GPU idle
		stagingActiveCount_get() == 0 && staging.size() > 1
			&& currentStagingInputs >= minInBatchSize
	)) ? currentStagingInputs : 0;

	if(req->submitInputs) {
		stagingActiveCount_inc();
		area.setIsActive(true);
		statBatchesStarted++;
		currentStagingInputs = 0;
		if(++currentStagingArea == staging.size())
			currentStagingArea = 0;
	}

	processingAdd = true;
	auto future = req->promPrep.get_future();
	transferThread.send(req);
	return future;
}

FUTURE_RETURN_T PAR2ProcMetal::addInput(const void* buffer, size_t size,
                                        uint16_t inputNum, bool flush) {
	return _addInput(buffer, size, inputNum, flush);
}

FUTURE_RETURN_T PAR2ProcMetal::addInput(const void* buffer, size_t size,
                                        const uint16_t* coeffs, bool flush) {
	return _addInput(buffer, size, coeffs, flush);
}

void PAR2ProcMetal::dummyInput(uint16_t inputNum, bool flush) {
	// Benchmarking hook: account for an input without transferring it.
	auto& area = staging[currentStagingArea];
	set_coeffs(area, currentStagingInputs, inputNum);
	currentStagingInputs++;
	unsigned submit = (flush || currentStagingInputs == inputBatchSize)
		? currentStagingInputs : 0;
	if(submit) {
		stagingActiveCount_inc();
		area.setIsActive(true);
		statBatchesStarted++;
		currentStagingInputs = 0;
		unsigned a = currentStagingArea;
		if(++currentStagingArea == staging.size()) currentStagingArea = 0;
		run_kernel(a, submit);
	}
	processingAdd = true;
}

bool PAR2ProcMetal::fillInput(const void* buffer) {
	(void)buffer;
	return false; // benchmarking hook, not needed by par2cmdline
}

void PAR2ProcMetal::flush() {
	if(!currentStagingInputs) return;

	MetalTransferReq* req = new MetalTransferReq();
	req->finish = false;
	req->parent = this;
	req->impl = impl.get();
	req->gf = gf.get();
	req->local = NULL; // flush marker: dispatch without staging anything
	req->area = currentStagingArea;
	req->submitInputs = currentStagingInputs;

	stagingActiveCount_inc();
	staging[currentStagingArea].setIsActive(true);
	statBatchesStarted++;
	currentStagingInputs = 0;
	if(++currentStagingArea == staging.size())
		currentStagingArea = 0;

	transferThread.send(req);
}

// ---- output --------------------------------------------------------------

FUTURE_RETURN_BOOL_T PAR2ProcMetal::getOutput(unsigned index, void* output) {
	MetalTransferReq* req = new MetalTransferReq();
	req->finish = true;
	req->parent = this;
	req->impl = impl.get();
	req->gf = gf.get();
	req->local = output;
	req->sliceLen = sliceSize;
	req->totalLen = sliceSizeAligned;
	req->outputIndex = index;
	req->outputOffset = (size_t)index * sliceSizeAligned;

	auto future = req->promOut.get_future();
	transferThread.send(req);
	return future;
}
