//  This file is part of par2cmdline (a PAR 2.0 compatible file verification and
//  repair tool). See http://parchive.sourceforge.net for details of PAR 2.0.
//
//  Copyright (c) 2003 Peter Brian Clements
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

#ifndef __PROGRESSTHROTTLE_H__
#define __PROGRESSTHROTTLE_H__

#include <chrono>

// Rate limiter for the progress lines that are written to the console.
//
// Progress is measured in tenths of a percent, so without throttling a
// progress line is written up to 1000 times per operation. That is far
// more often than anyone can read it and, on a slow or redirected
// terminal, the writing can cost more than the work being reported on.
//
// Ready() only allows one update through per interval, so the display is
// refreshed at a fixed rate no matter how quickly the underlying value
// changes. Pass force to let an update through regardless of the timing:
// callers use it for the final update of an operation so that the last
// line written is always the completed one.
//
// The object is not thread safe. Callers that write progress from more
// than one thread must call Ready() from inside the critical section that
// already serialises their output.

class ProgressThrottle
{
public:
  explicit ProgressThrottle(unsigned int intervalms = 100)
  : interval(std::chrono::milliseconds(intervalms))
  , last()
  , started(false)
  {
  }

  // Should a progress line be written now?
  bool Ready(bool force = false)
  {
    std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();

    if (started && !force && now - last < interval)
      return false;

    last = now;
    started = true;
    return true;
  }

private:
  std::chrono::steady_clock::duration   interval; // Minimum time between updates
  std::chrono::steady_clock::time_point last;     // When the last update was allowed
  bool                                  started;  // Whether "last" holds a real time
};

#endif // __PROGRESSTHROTTLE_H__
