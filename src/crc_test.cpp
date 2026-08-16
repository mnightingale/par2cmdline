//  This file is part of par2cmdline (a PAR 2.0 compatible file verification and
//  repair tool). See http://parchive.sourceforge.net for details of PAR 2.0.
//
//  Copyright (c) 2019 Michael D. Nahas
//
//  par2cmdline is free software; you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation; either version 2 of the License, or
//  (at your option) any later version.
//
//  par2cmdline is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with this program; if not, write to the Free Software
//  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA


// This is just a simple set of tests on md5 hash.
// The initial version just showed it was self-consistent, not accurate.

// I compile with:
//    g++ -DHAVE_CONFIG_H -I.. crc_test.cpp crc.cpp


#include "libpar2internal.h"

#include <iostream>
#include <stdlib.h>

#include "crc.h"

// The implementations themselves, so this test can cover the ones dispatch
// did not select.
#include "crc_slice4.h"
#include "crc_arm.h"
#include "crc_clmul.h"


// Example usage:
//   u32 checksum = ~0 ^ CRCUpdateBlock(~0, (size_t)blocksize, buffer);


// compares UpdateBlock(crc, length) to UpdateBlock(crc,buffer,buffersize)
int test1() {
  unsigned char buffer[] = {0,0,0,0,0,0,0,0};

  u32 checksum1 = ~0 ^ CRCUpdateBlock(~0, sizeof(buffer), buffer);
  u32 checksum2 = ~0 ^ CRCUpdateBlock(~0, sizeof(buffer));

  if (checksum1 != checksum2) {
    std::cerr << "checksum1 = " << checksum1 << std::endl;
    std::cerr << "checksum2 = " << checksum2 << std::endl;
    return 1;
  }

  return 0;
}


// CRC32 of "123456789" yields 0xCBF43926
// according to http://www.ross.net/crc/download/crc_v3.txt
int test2() {
  unsigned char buffer[] = "123456789";
  size_t buffer_length = 9;
  u32 expected_checksum = 0xCBF43926u;

  u32 checksum1 = ~0 ^ CRCUpdateBlock(~0, buffer_length, buffer);

  if (checksum1 != expected_checksum) {
    std::cerr << "checksum was not precalculated value: " << std::hex << checksum1 << std::dec << std::endl;
    std::cerr << "   expected " << std::hex << expected_checksum << std::dec << std::endl;
    return 1;
  }

  return 0;
}


// generate random data.
// put it into checksum using different length blocks
// make sure output is the same.
int test3() {
  srand(345087209);
  unsigned char buffer[32*1024];

  for (unsigned int i = 0; i < sizeof(buffer); i++) {
    buffer[i] = (unsigned char) (rand() % 256);
  }

  u32 checksum1 = ~0;
  unsigned int offset = 0;
  while (offset < sizeof(buffer)) {
    unsigned int length = (unsigned int) (rand() % 256);
    if (offset + length > sizeof(buffer))
      length = sizeof(buffer) - offset;
    checksum1 = CRCUpdateBlock(checksum1, length, buffer + offset);
    offset += length;
  }
  checksum1 = ~0 ^ checksum1;


  u32 checksum2 = ~0;
  offset = 0;
  while (offset < sizeof(buffer)) {
    unsigned int length = (unsigned int) (rand() % 256);
    if (offset + length > sizeof(buffer))
      length = sizeof(buffer) - offset;
    checksum2 = CRCUpdateBlock(checksum2, length, buffer + offset);
    offset += length;
  }
  checksum2 = ~0 ^ checksum2;

  if (checksum1 != checksum2) {
    std::cerr << "random checksum1 = " << checksum1 << std::endl;
    std::cerr << "random checksum2 = " << checksum2 << std::endl;
    return 1;
  }

  return 0;
}


// generate random data.
// compare char-at-a-time vs block
// make sure output is the same.
int test4() {
  srand(113450911);
  unsigned char buffer[32*1024];

  for (unsigned int i = 0; i < sizeof(buffer); i++) {
    buffer[i] = (unsigned char) (rand() % 256);
  }

  u32 checksum1 = ~0;
  unsigned int offset = 0;
  while (offset < sizeof(buffer)) {
    unsigned int length = (int) (rand() % 256);
    if (offset + length > sizeof(buffer))
      length = sizeof(buffer) - offset;
    checksum1 = CRCUpdateBlock(checksum1, length, buffer + offset);
    offset += length;
  }
  checksum1 = ~0 ^ checksum1;


  u32 checksum2 = ~0;
  for (offset = 0; offset < sizeof(buffer); offset++) {
    checksum2 = CRCUpdateChar(checksum2, *(buffer + offset));
  }
  checksum2 = ~0 ^ checksum2;

  if (checksum1 != checksum2) {
    std::cerr << "random checksum1 = " << checksum1 << std::endl;
    std::cerr << "random checksum2 = " << checksum2 << std::endl;
    return 1;
  }

  return 0;
}



// check windowing on random buffer
int test5() {
  srand(113450911);
  unsigned char buffer[32*1024];

  for (unsigned int i = 0; i < sizeof(buffer); i++) {
    buffer[i] = (unsigned char) (rand() % 256);
  }

  u64 window = 1024;

  u32 windowtable[256];
  GenerateWindowTable(window, windowtable);

  int result = 0;

  u32 crc = ~0 ^ CRCUpdateBlock(~0, window, buffer);
  for (int offset = 0; offset + window < sizeof(buffer) - 1; offset++) {
    // compare against reference
    u32 othercrc = ~0 ^ CRCUpdateBlock(~0, window, buffer + offset);
    if (crc != othercrc) {
      std::cerr << "error in window at offset " << offset << std::endl;
      std::cerr << "  checksum1 = " << crc << std::endl;
      std::cerr << "  checksum2 = " << othercrc << std::endl;
      result = 1;
    }

    // slide window
    crc = CRCSlideChar(crc, buffer[offset + window], buffer[offset], windowtable);
  }

  return result;
}


// Checksum of checksum table
// stolen from:
// http://www.efg2.com/Lab/Mathematics/CRC.htm
int test6() {
  u32 checksum1 = ~0 ^ CRCUpdateBlock(~0, sizeof(ccitttable.table), &ccitttable.table);
  u32 expected = 0x6FCF9E13;
  if (checksum1 != expected) {
      std::cerr << "error when computing checksum of checksum table " << std::endl;
      std::cerr << "  checksum1 = " << checksum1 << std::endl;
      std::cerr << "  expected = " << expected << std::endl;
      return 0;
  }

  return 0;
}





// Byte-at-a-time reference for the block implementations to match.
u32 CRCReference(u32 crc, size_t length, const void *buffer) {
  const u8 *current = (const u8 *)buffer;
  while (length-- > 0)
    crc = CRCUpdateChar(crc, *current++);
  return crc;
}


// Every implementation built for this CPU must agree with the reference,
// including the ones dispatch did not select.
int test7() {
  struct implementation {
    const char *name;
    u32 (*fn)(u32, size_t, const void*);
  };
  implementation impls[3];
  unsigned count = 0;

  // The tables are per translation unit, so this one needs its own built.
  BuildSliceTables();
  impls[count].name = "slice4";
  impls[count++].fn = &CRCUpdateBlock_Slice4;
#ifdef PAR2_CRC_X86
  if (X86HasPclMul()) {
    impls[count].name = "pclmul";
    impls[count++].fn = &CRCUpdateBlock_PclMul;
# ifdef PAR2_CRC_X86_VPCLMUL
    if (X86HasVPclMul()) {
      impls[count].name = "vpclmul";
      impls[count++].fn = &CRCUpdateBlock_VPclMul;
    }
# endif
  }
#endif
#ifdef PAR2_CRC_ARM
  if (ArmHasCRC()) {
    impls[count].name = "armcrc";
    impls[count++].fn = &CRCUpdateBlock_ArmCRC;
  }
#endif

  const size_t buffer_length = 8192;
  u8 buffer[buffer_length];
  for (size_t i = 0; i < buffer_length; i++)
    buffer[i] = (u8)(i * 37 + (i >> 5));

  // lengths either side of every stride an implementation may use, plus tails
  const size_t lengths[] = {0, 1, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 47, 48,
                            63, 64, 65, 79, 80, 95, 96, 112, 127, 128, 129,
                            143, 144, 191, 192, 255, 256, 257, 383, 384, 511,
                            512, 1000, 4095, 4096, 8192};

  for (unsigned li = 0; li < sizeof(lengths)/sizeof(lengths[0]); li++) {
    const size_t length = lengths[li];
    for (size_t offset = 0; offset < 8 && offset + length <= buffer_length; offset++) {
      const u32 expected = CRCReference(~0, length, buffer + offset);
      for (unsigned impl = 0; impl < count; impl++) {
        const u32 crc = impls[impl].fn(~0, length, buffer + offset);
        if (crc != expected) {
          std::cerr << "CRC mismatch: " << impls[impl].name
                    << " length " << length << " offset " << offset
                    << " got " << crc << " expected " << expected << std::endl;
          return 1;
        }
      }
    }
  }

  return 0;
}

int main() {
  if (test1()) {
    std::cerr << "FAILED: test1" << std::endl;
    return 1;
  }
  if (test2()) {
    std::cerr << "FAILED: test2" << std::endl;
    return 1;
  }
  if (test3()) {
    std::cerr << "FAILED: test3" << std::endl;
    return 1;
  }
  if (test4()) {
    std::cerr << "FAILED: test4" << std::endl;
    return 1;
  }
  if (test5()) {
    std::cerr << "FAILED: test5" << std::endl;
    return 1;
  }
  if (test6()) {
    std::cerr << "FAILED: test6" << std::endl;
    return 1;
  }
  if (test7()) {
    std::cerr << "FAILED: test7" << std::endl;
    return 1;
  }

  std::cout << "SUCCESS: crc_test complete." << std::endl;

  return 0;
}
