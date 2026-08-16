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

#ifndef __CRC_SLICE4_H__
#define __CRC_SLICE4_H__

// Table-driven CRC32, four bytes at a time. Included by crc.cpp.

// Four tables, one per byte of the word being consumed.
static u32 crcslice[4][256];

static void BuildSliceTables()
{
  for (u32 i = 0; i < 256; i++)
    crcslice[0][i] = ccitttable.table[i];

  for (u32 i = 0; i < 256; i++)
  {
    u32 c = crcslice[0][i];
    for (int t = 1; t < 4; t++)
    {
      c = crcslice[0][c & 0xff] ^ (c >> 8);
      crcslice[t][i] = c;
    }
  }
}

static u32 CRCUpdateBlock_Slice4(u32 crc, size_t length, const void *buffer)
{
  const unsigned char *current = (const unsigned char *)buffer;

  while (length >= 4)
  {
    const u32 a = crc ^ ((u32)current[0]        | ((u32)current[1] << 8)
                      | ((u32)current[2] << 16) | ((u32)current[3] << 24));
    crc = crcslice[3][ a        & 0xff] ^ crcslice[2][(a >>  8) & 0xff]
        ^ crcslice[1][(a >> 16) & 0xff] ^ crcslice[0][(a >> 24) & 0xff];
    current += 4;
    length -= 4;
  }

  while (length-- > 0)
    crc = ((crc >> 8) & 0x00ffffffL) ^ ccitttable.table[(u8)crc ^ (*current++)];

  return crc;
}

#endif // __CRC_SLICE4_H__
