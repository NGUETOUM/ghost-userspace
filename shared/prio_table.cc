// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "shared/prio_table.h"

#include <cstdint>

namespace ghost {

// Warning insanity requires "constexpr const" here.
static constexpr const char* kPrioTableShmemName = "priotable";
static constexpr int64_t kPrioTableVersion = 0;

static size_t shmem_size(uint32_t sched_items, uint32_t work_classes,
                         uint32_t stream_capacity) {
  size_t sz = 0;

  sz += sizeof(struct legorch_shmem_hdr);
  sz += sizeof(struct sched_item) * sched_items;
  sz += sizeof(struct work_class) * work_classes;
  // Check that 'sz' is a multiple of the cacheline size so that the stream
  // starts on a new cacheline
  // The three structs above are each aligned to a cacheline, so this check
  // should succeed
  CHECK_ZERO(sz % ABSL_CACHELINE_SIZE);
  sz += sizeof(struct ghost::PrioTable::stream) +
        sizeof(std::atomic<int>) * stream_capacity;

  return sz;
}

PrioTable::PrioTable(uint32_t num_items, uint32_t num_classes,
                     StreamCapacity stream_capacity) {
  uint32_t st_cap =
      static_cast<std::underlying_type<StreamCapacity>::type>(stream_capacity);
  size_t size = shmem_size(num_items, num_classes, st_cap);
  shmem_ = std::make_unique<GhostShmem>(kPrioTableVersion, kPrioTableShmemName,
                                        size);
  hdr_ = reinterpret_cast<struct legorch_shmem_hdr*>(shmem_->bytes());

  hdr()->version = 0;
  hdr()->hdrlen = sizeof(struct legorch_shmem_hdr);
  hdr()->maplen = size;
  hdr()->si_num = num_items;
  hdr()->si_off = hdr()->hdrlen;
  hdr()->wc_num = num_classes;
  hdr()->wc_off = hdr()->si_off + hdr()->si_num * sizeof(struct sched_item);
  hdr()->st_cap = st_cap;
  hdr()->st_off = hdr()->wc_off + hdr()->wc_num * sizeof(struct work_class);
  // Check that the stream starts on an address aligned to the cacheline size
  // The header, sched items, and work classes are each aligned to a cacheline,
  // so this check should succeed
  CHECK_ZERO(hdr()->st_off % ABSL_CACHELINE_SIZE);

  std::atomic<int>* entries = stream()->entries;
  for (uint32_t i = 0; i < hdr()->st_cap; i++) {
    entries[i].store(kStreamFreeEntry, std::memory_order_relaxed);
  }

  shmem_->MarkReady();  // Ready for ghOSt agent to connect/start polling.
}

bool PrioTable::Attach(pid_t remote) {
  shmem_ = std::make_unique<GhostShmem>();
  const bool ret =
      shmem_->Attach(kPrioTableVersion, kPrioTableShmemName, remote);
  if (!ret) {
    return ret;
  }
  hdr_ = reinterpret_cast<struct legorch_shmem_hdr*>(shmem_->bytes());
  CHECK_GE(shmem_->size(), hdr_->maplen);  // Paranoia.
  return true;
}

bool PrioTable::Attach(pid_t remote, int fd) {
  shmem_ = std::make_unique<GhostShmem>();
  const bool ret =
      shmem_->Attach(kPrioTableVersion, kPrioTableShmemName, remote, fd);
  if (!ret) {
    return ret;
  }
  hdr_ = reinterpret_cast<struct legorch_shmem_hdr*>(shmem_->bytes());
  CHECK_GE(shmem_->size(), hdr_->maplen);  // Paranoia.
  return true;
}

PrioTable::~PrioTable() {}

struct ghost::PrioTable::stream* PrioTable::stream() {
  char* bytes = reinterpret_cast<char*>(hdr_);
  return reinterpret_cast<struct ghost::PrioTable::stream*>(bytes +
                                                            hdr()->st_off);
}

void PrioTable::MarkUpdatedIndex(int idx, int num_retries) {
  struct stream* s = stream();
  std::atomic<int>* scrape_all = &s->scrape_all;
  std::atomic<int>* entries = s->entries;

  // Already in overflow? Ensure we are covered by a scrape_all pass.
  if (scrape_all->load(std::memory_order_relaxed) > 0) {
    scrape_all->fetch_add(1, std::memory_order_release);
    return;
  }

  for (int i = 0; i < num_retries + 1; i++) {
    std::atomic<int>* cell = &entries[(idx + i) % hdr()->st_cap];
    int expected = kStreamFreeEntry;
    if (cell->load(std::memory_order_relaxed) == kStreamFreeEntry &&
        cell->compare_exchange_weak(expected, idx, std::memory_order_release,
                                    std::memory_order_relaxed)) {
      return;
    }
  }

  scrape_all->fetch_add(1, std::memory_order_release);
}

// Returns the index of the next updated element.
// kStreamNoEntries         : no updated entries
// kStreamOverflow          : overflow, all entries potentially updated
// [0, hdr()->si_num - 1]   : entry at returned index
int PrioTable::NextUpdatedIndex() {
  struct stream* s = stream();
  std::atomic<int>* scrape_all = &s->scrape_all;
  std::atomic<int>* entries = s->entries;
  bool full_scan = false;

  if (scrape_all->load(std::memory_order_relaxed) > 0) {
    scrape_all->exchange(0, std::memory_order_acquire);
    full_scan = true;
  }

  for (uint32_t i = 0; i < hdr()->st_cap; i++) {
    // We always need to also acquire versus our advertised indices below as
    // they may represent writes (prior to capacity) not paired with
    // scrape_all.
    int idx = entries[i].load(std::memory_order_acquire);
    if (idx != kStreamFreeEntry) {
      // This is slightly subtle, a non-atomic RMW (versus exchange) is safe as
      // we are guaranteed that only a single writer can proceed (due to
      // cmpxchg). In the case we race on not observing an ongoing writer, we
      // must see them (and their updates) in a future NextUpdatedIndex() call.
      entries[i].store(kStreamFreeEntry, std::memory_order_relaxed);
      if (!full_scan) return idx;
    }
  }

  return full_scan ? kStreamOverflow : kStreamNoEntries;
}

}  // namespace ghost
