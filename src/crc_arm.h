//  This file is part of par2cmdline (a PAR 2.0 compatible file verification and
//  repair tool). See http://parchive.sourceforge.net for details of PAR 2.0.
//
//  Copyright (c) 2003 Peter Brian Clements
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

#ifndef __CRC_ARM_H__
#define __CRC_ARM_H__

// CRC32 using the ARMv8 CRC extension. Included by crc.cpp.

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__ARM_ARCH_8A)
# define PAR2_CRC_ARM 1
# include <arm_acle.h>
# if defined(__APPLE__)
#  include <sys/sysctl.h>
# elif defined(__linux__)
#  include <sys/auxv.h>
#  ifndef HWCAP_CRC32
#   define HWCAP_CRC32 (1 << 7)
#  endif
# endif
#endif

#ifdef PAR2_CRC_ARM

#ifndef PAR2_TARGET
# ifdef __GNUC__
#  define PAR2_TARGET(isa) __attribute__((target(isa)))
# else
#  define PAR2_TARGET(isa)
# endif
#endif

PAR2_TARGET("crc")
static u32 CRCUpdateBlock_ArmCRC(u32 crc, size_t length, const void *buffer)
{
  const unsigned char *current = (const unsigned char *)buffer;

  while (length >= 32)
  {
    u64 v0, v1, v2, v3;
    memcpy(&v0, current,      8);
    memcpy(&v1, current +  8, 8);
    memcpy(&v2, current + 16, 8);
    memcpy(&v3, current + 24, 8);
    crc = __crc32d(crc, v0);
    crc = __crc32d(crc, v1);
    crc = __crc32d(crc, v2);
    crc = __crc32d(crc, v3);
    current += 32;
    length -= 32;
  }
  while (length >= 8)
  {
    u64 v;
    memcpy(&v, current, 8);
    crc = __crc32d(crc, v);
    current += 8;
    length -= 8;
  }
  while (length-- > 0)
    crc = __crc32b(crc, *current++);

  return crc;
}

static bool ArmHasCRC()
{
# if defined(__APPLE__)
  int val = 0;
  size_t len = sizeof(val);
  if (sysctlbyname("hw.optional.armv8_crc32", &val, &len, NULL, 0) == 0)
    return val != 0;
  return false;
# elif defined(__linux__)
  return (getauxval(AT_HWCAP) & HWCAP_CRC32) != 0;
# elif defined(_WIN32)
  return IsProcessorFeaturePresent(PF_ARM_V8_CRC32_INSTRUCTIONS_AVAILABLE) != 0;
# else
  return false;
# endif
}

#endif // PAR2_CRC_ARM

#endif // __CRC_ARM_H__
