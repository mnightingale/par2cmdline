// GF(2^16) multiply-add kernel for PAR2 recovery computation.
//
// Computes, for each output slice o and input slice i:
//     dst[o] ^= gf16_mul(src[i], coeff[o][i])
//
// Multiplication uses precomputed nibble lookup tables rather than log/antilog
// tables. GF16 multiplication is linear over GF(2) in its second operand, so
// for a fixed coefficient c a value v splits into four nibbles:
//
//     c * v = XOR over j of (c * (nibble_j << 4j))
//
// That gives four 16-entry tables per coefficient - 128 bytes - against the
// 128 KiB a full antilog table would need. This matters because Apple GPUs
// expose only 32 KiB of threadgroup memory, so a full table could not be
// cached there at all, while these fit comfortably for a whole output group.
//
// The tables are built host-side (see gf16_metal_build_luts) because the cost
// is trivial next to the data volume, and building them on the GPU would have
// to be repeated by every threadgroup.

#include <metal_stdlib>
using namespace metal;

// Upper bound on outputs handled by one threadgroup. Bounds the accumulator
// array so it stays in registers rather than spilling to stack.
#define GF16_MAX_OUTPUTS_PER_GROUP 8

// Entries (ushorts) in one coefficient's lookup table: 4 tables x 16 entries.
#define GF16_LUT_ENTRIES 64

struct GF16Params {
	uint numInputs;       // input slices in this batch
	uint numOutputs;      // total output slices
	uint outputsPerGroup; // outputs handled per threadgroup (<= MAX)
	uint vecPerSlice;     // uint4 units of live data per slice
	uint sliceStrideVec;  // uint4 units between consecutive slices
	uint accumulate;      // 1 = XOR into dst, 0 = overwrite
};

// One GF16 multiply via the nibble tables: 4 lookups, 3 XORs.
static inline ushort lut_mul(threadgroup const ushort* L, ushort v) {
	return L[      (v      ) & 0xF]
	     ^ L[16 + ((v >>  4) & 0xF)]
	     ^ L[32 + ((v >>  8) & 0xF)]
	     ^ L[48 + ((v >> 12) & 0xF)];
}

// Eight GF16 values packed into a uint4, for coalesced 16-byte accesses.
static inline uint4 lut_mul4(threadgroup const ushort* L, uint4 v) {
	uint4 r;
#pragma unroll
	for(uint k = 0; k < 4; k++) {
		uint w = v[k];
		ushort lo = lut_mul(L, (ushort)(w & 0xFFFF));
		ushort hi = lut_mul(L, (ushort)(w >> 16));
		r[k] = (uint)lo | ((uint)hi << 16);
	}
	return r;
}

kernel void gf16_muladd(
	device uint4*         dst    [[buffer(0)]],
	device const uint4*   src    [[buffer(1)]],
	device const ushort*  luts   [[buffer(2)]],
	constant GF16Params&  p      [[buffer(3)]],
	threadgroup ushort*   lut    [[threadgroup(0)]],
	uint3 tgpos   [[threadgroup_position_in_grid]],
	uint3 tgdim   [[threadgroups_per_grid]],
	uint3 tgdim3  [[threads_per_threadgroup]],
	uint  tid     [[thread_index_in_threadgroup]])
{
	const uint tgsize = tgdim3.x;

	const uint outBase = tgpos.y * p.outputsPerGroup;
	if(outBase >= p.numOutputs) return;
	const uint nOut = min(p.outputsPerGroup, p.numOutputs - outBase);

	// Stage this output group's tables into threadgroup memory once. Every
	// column iteration below reads them, so the copy pays for itself as soon
	// as the group covers more than a handful of columns.
	const uint lutCount = nOut * p.numInputs * GF16_LUT_ENTRIES;
	const uint lutBase = outBase * p.numInputs * GF16_LUT_ENTRIES;
	for(uint i = tid; i < lutCount; i += tgsize)
		lut[i] = luts[lutBase + i];
	threadgroup_barrier(mem_flags::mem_threadgroup);

	// Grid-stride over the slice so a fixed dispatch covers any slice size.
	const uint stride = tgdim.x * tgsize;
	for(uint col = tgpos.x * tgsize + tid; col < p.vecPerSlice; col += stride) {
		uint4 acc[GF16_MAX_OUTPUTS_PER_GROUP];
#pragma unroll
		for(uint o = 0; o < GF16_MAX_OUTPUTS_PER_GROUP; o++)
			acc[o] = uint4(0);

		// Each input is read once and applied to every output in the group,
		// which is what keeps this compute-bound rather than bandwidth-bound.
		for(uint i = 0; i < p.numInputs; i++) {
			const uint4 v = src[i * p.sliceStrideVec + col];
			for(uint o = 0; o < nOut; o++) {
				threadgroup const ushort* L =
					lut + (o * p.numInputs + i) * GF16_LUT_ENTRIES;
				acc[o] ^= lut_mul4(L, v);
			}
		}

		for(uint o = 0; o < nOut; o++) {
			const uint idx = (outBase + o) * p.sliceStrideVec + col;
			dst[idx] = p.accumulate ? (dst[idx] ^ acc[o]) : acc[o];
		}
	}
}
