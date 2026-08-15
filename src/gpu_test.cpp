// Verifies the GPU GF16 backend against the CPU backend through the full
// PAR2Proc interface: staging, batching, checksum-carrying transfers and
// readback, not just the raw kernel.
//
// Skips (exits 0) when no usable GPU is present, so `make check` passes on
// machines without one.

#include "libpar2internal.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

#include "../parpar/gf16/controller.h"
#include "../parpar/gf16/controller_cpu.h"
#include "../parpar/gf16/gpu_device.h"
#include "../parpar/gf16/gfmat_coeff.h"

#ifdef PARPAR_METAL_SUPPORT
# include "../parpar/gf16/controller_metal.h"
#endif
#ifdef PARPAR_VULKAN_SUPPORT
# include "../parpar/gf16/controller_vulkan.h"
#endif

// Which backend this build tests. The two are mutually exclusive in practice --
// Metal is macOS-only, Vulkan is everywhere else -- so choosing at compile time
// is enough, and it keeps the test a straight A/B against the CPU backend.
#if defined(PARPAR_METAL_SUPPORT)
typedef PAR2ProcMetal GpuBackend;
# define GPU_BACKEND_NAME "Metal"
#elif defined(PARPAR_VULKAN_SUPPORT)
typedef PAR2ProcVulkan GpuBackend;
# define GPU_BACKEND_NAME "Vulkan"
#endif

#if defined(PARPAR_METAL_SUPPORT) || defined(PARPAR_VULKAN_SUPPORT)
# define HAVE_GPU_BACKEND 1
#endif

// The two ways par2cmdline drives a backend. Creation derives coefficients
// from the input's block number; repair supplies them directly from the
// inverted Reed-Solomon matrix. Both must be exercised - they take different
// set_coeffs paths.
enum CoeffMode { COEFF_FROM_INPUT_NUM, COEFF_EXPLICIT };

// Runs one pass and returns every output slice concatenated.
template<class Backend>
static bool run_backend(Backend& backend, PAR2Proc& proc,
                        unsigned numInputs, unsigned numOutputs,
                        size_t sliceSize, const std::vector<char>& inputs,
                        std::vector<char>& outputs, std::ostream& err,
                        CoeffMode mode, const std::vector<uint16_t>& explicitCoeffs)
{
	if(!proc.init(sliceSize, {{&backend, 0, sliceSize}})) {
		err << "init failed" << std::endl;
		return false;
	}
	if(!proc.setRecoverySlices(numOutputs)) {
		err << "setRecoverySlices failed" << std::endl;
		return false;
	}
	if(!proc.setCurrentSliceSize(sliceSize)) {
		err << "setCurrentSliceSize failed" << std::endl;
		return false;
	}

	proc.discardOutput();
	for(unsigned i = 0; i < numInputs; i++) {
		proc.waitForAdd();
		const void* buf = inputs.data() + (size_t)i * sliceSize;
		const bool last = (i + 1 == numInputs);
		auto f = (mode == COEFF_EXPLICIT)
			? proc.addInput(buf, sliceSize,
			                explicitCoeffs.data() + (size_t)i * numOutputs, last)
			: proc.addInput(buf, sliceSize, (uint16_t)i, last);
		f.get();
	}
	proc.endInput().get();

	outputs.assign((size_t)numOutputs * sliceSize, 0);
	for(unsigned o = 0; o < numOutputs; o++) {
		auto f = proc.getOutput(o, outputs.data() + (size_t)o * sliceSize);
		if(!f.get()) {
			err << "output " << o << " failed its internal checksum" << std::endl;
			return false;
		}
	}
	return true;
}

int main(void)
{
	gfmat_init();

#ifndef HAVE_GPU_BACKEND
	std::cout << "SKIP: built without GPU backend support" << std::endl;
	return 0;
#else
	std::vector<GPUDeviceInfo> devices = gpu_enumerate_devices();
	if(gpu_default_device() < 0) {
		std::cout << "SKIP: no usable GPU device" << std::endl;
		return 0;
	}
	std::cout << "Testing " GPU_BACKEND_NAME " against: "
		<< devices[gpu_default_device()].name << std::endl;

	std::mt19937 rng(20260814);

	// Shapes chosen to exercise the batching edges: input counts below, equal
	// to and above the staging batch size; output counts that do and do not
	// divide the threadgroup output grouping; and slice sizes that are not
	// multiples of the kernel's 16-byte vector.
	struct Shape { unsigned inputs, outputs; size_t slice; };
	const Shape shapes[] = {
		{1,  1,   4096},
		{3,  2,   4096},
		{8,  5,   8192},
		{16, 4,   4096},
		{17, 3,   4096},
		{33, 7,   2048},
		{5,  9,   4096},
		{4,  3,   1020},  // not a multiple of 16
		{4,  3,     16},  // single vector
		{2,  1,      4},  // smaller than one vector
	};

	unsigned failures = 0;
	for(const Shape& s : shapes)
	for(int modeIdx = 0; modeIdx < 2; modeIdx++) {
		const CoeffMode mode = modeIdx ? COEFF_EXPLICIT : COEFF_FROM_INPUT_NUM;
		const char* modeName = modeIdx ? "explicit" : "blocknum";

		std::vector<char> inputs((size_t)s.inputs * s.slice);
		for(auto& c : inputs) c = (char)rng();

		// Arbitrary non-zero coefficients, standing in for an inverted RS matrix.
		std::vector<uint16_t> coeffs((size_t)s.inputs * s.outputs);
		for(auto& c : coeffs) c = (uint16_t)(rng() | 1);

		std::vector<char> cpuOut, gpuOut;

		PAR2ProcCPU cpuBackend;
		cpuBackend.setSliceSize(s.slice);
		if(!cpuBackend.init(GF16_AUTO)) {
			std::cerr << "CPU backend init failed" << std::endl;
			return 1;
		}
		PAR2Proc cpuProc;
		if(!run_backend(cpuBackend, cpuProc, s.inputs, s.outputs, s.slice,
		                inputs, cpuOut, std::cerr, mode, coeffs)) {
			std::cerr << "CPU reference run failed" << std::endl;
			return 1;
		}
		cpuProc.deinit();

		GpuBackend gpuBackend;
		if(!gpuBackend.isAvailable()) {
			std::cout << "SKIP: GPU backend unavailable" << std::endl;
			return 0;
		}
		gpuBackend.setSliceSize(s.slice);
		if(!gpuBackend.init()) {
			std::cerr << "GPU backend init failed" << std::endl;
			return 1;
		}
		PAR2Proc gpuProc;
		if(!run_backend(gpuBackend, gpuProc, s.inputs, s.outputs, s.slice,
		                inputs, gpuOut, std::cerr, mode, coeffs)) {
			std::cerr << "GPU run failed for in=" << s.inputs
				<< " out=" << s.outputs << " slice=" << s.slice
				<< " mode=" << modeName << std::endl;
			failures++;
			continue;
		}
		gpuProc.deinit();

		if(cpuOut.size() != gpuOut.size()) {
			std::cerr << "FAIL size mismatch" << std::endl;
			failures++;
			continue;
		}
		size_t bad = 0, firstBad = 0;
		for(size_t i = 0; i < cpuOut.size(); i++) {
			if(cpuOut[i] != gpuOut[i]) {
				if(!bad) firstBad = i;
				bad++;
			}
		}
		if(bad) {
			std::cerr << "FAIL in=" << s.inputs << " out=" << s.outputs
				<< " slice=" << s.slice << " mode=" << modeName
				<< ": " << bad << '/' << cpuOut.size()
				<< " bytes differ, first at " << firstBad << std::endl;
			failures++;
		} else {
			std::cout << "  ok  in=" << s.inputs << " out=" << s.outputs
				<< " slice=" << s.slice << " mode=" << modeName << std::endl;
		}
	}

	if(failures) {
		std::cerr << "ERROR: " << failures << " shape(s) mismatched the CPU backend"
			<< std::endl;
		return 1;
	}
	std::cout << "SUCCESS: gpu_test complete." << std::endl;
	return 0;
#endif
}
