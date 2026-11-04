// SPDX-FileCopyrightText: 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <iterator>
#include <boost/icl/discrete_interval.hpp>
#include <boost/icl/interval_map.hpp>
#include <boost/icl/split_interval_map.hpp>
#include <boost/icl/split_interval_set.hpp>
#include <boost/pool/pool.hpp>
#include <boost/pool/pool_alloc.hpp>
#include <boost/pool/poolfwd.hpp>
#include "common/types.h"

namespace VideoCore {

template <class T>
using RangeSetsAllocator =
    boost::fast_pool_allocator<T, boost::default_user_allocator_new_delete,
                               boost::details::pool::default_mutex, 1024, 2048>;

struct RangeSet {
    using IntervalSet =
        boost::icl::interval_set<VAddr, std::less,
                                 ICL_INTERVAL_INSTANCE(ICL_INTERVAL_DEFAULT, VAddr, std::less),
                                 RangeSetsAllocator>;
    using IntervalType = typename IntervalSet::interval_type;

    explicit RangeSet() = default;
    ~RangeSet() = default;

    void Add(VAddr base_address, size_t size) {
        const VAddr end_address = base_address + size;
        IntervalType interval{base_address, end_address};
        m_ranges_set.add(interval);
    }

    void Subtract(VAddr base_address, size_t size) {
        const VAddr end_address = base_address + size;
        IntervalType interval{base_address, end_address};
        m_ranges_set.subtract(interval);
    }

    void Clear() {
        m_ranges_set.clear();
    }

    bool Contains(VAddr base_address, size_t size) const {
        const VAddr end_address = base_address + size;
        IntervalType interval{base_address, end_address};
        return boost::icl::contains(m_ranges_set, interval);
    }

    bool Intersects(VAddr base_address, size_t size) const {
        const VAddr end_address = base_address + size;
        IntervalType interval{base_address, end_address};
        return boost::icl::intersects(m_ranges_set, interval);
    }

    template <typename Func>
    void ForEach(Func&& func) const {
        if (m_ranges_set.empty()) {
            return;
        }

        for (const auto& set : m_ranges_set) {
            const VAddr inter_addr_end = set.upper();
            const VAddr inter_addr = set.lower();
            func(inter_addr, inter_addr_end);
        }
    }

    /// fifa-hybrid diagnostics: every interval overlapping [base, base+size), by a linear scan.
    template <typename Func>
    void ForEachOverlappingLinear(VAddr base_addr, size_t size, Func&& func) const {
        for (const auto& iv : m_ranges_set) {
            if (iv.lower() < base_addr + size && iv.upper() > base_addr) {
                func(iv.lower(), iv.upper());
            }
        }
    }

    template <typename Func>
    void ForEachInRange(VAddr base_addr, size_t size, Func&& func) const {
        if (m_ranges_set.empty()) {
            return;
        }
        const VAddr start_address = base_addr;
        const VAddr end_address = start_address + size;
        // fifa-hybrid: lower_bound/upper_bound with the whole query interval skipped overlapping
        // intervals (boost 1.88, 64-bit domain): a GPU-written range starting before a page run,
        // or the second range inside it, was never written back and then dropped (FIFA's type
        // table, DMA-copied at match start, came back as garbage and a culling shader looped for
        // a second per frame). Start from the interval around the first byte and walk while the
        // intervals begin before the end (verified against a brute-force reference).
        auto it = m_ranges_set.lower_bound(IntervalType{start_address, start_address + 1});
        if (it != m_ranges_set.begin()) {
            const auto prev = std::prev(it);
            if (prev->upper() > start_address) {
                it = prev;
            }
        }
        for (; it != m_ranges_set.end() && it->lower() < end_address; ++it) {
            if (it->upper() <= start_address) {
                continue;
            }
            func(std::max<VAddr>(it->lower(), start_address),
                 std::min<VAddr>(it->upper(), end_address));
        }
    }

    template <typename Func>
    void ForEachNotInRange(VAddr base_addr, size_t size, Func&& func) const {
        const VAddr end_addr = base_addr + size;
        ForEachInRange(base_addr, size, [&](VAddr range_addr, VAddr range_end) {
            if (size_t gap_size = range_addr - base_addr; gap_size != 0) {
                func(base_addr, gap_size);
            }
            base_addr = range_end;
        });
        if (base_addr != end_addr) {
            func(base_addr, end_addr - base_addr);
        }
    }

    IntervalSet m_ranges_set;
};

template <typename T>
class RangeMap {
public:
    using IntervalMap =
        boost::icl::interval_map<VAddr, T, boost::icl::total_absorber, std::less,
                                 boost::icl::inplace_identity, boost::icl::inter_section,
                                 ICL_INTERVAL_INSTANCE(ICL_INTERVAL_DEFAULT, VAddr, std::less),
                                 RangeSetsAllocator>;
    using IntervalType = typename IntervalMap::interval_type;

public:
    RangeMap() = default;
    ~RangeMap() = default;

    RangeMap(RangeMap const&) = delete;
    RangeMap& operator=(RangeMap const&) = delete;

    RangeMap(RangeMap&& other);
    RangeMap& operator=(RangeMap&& other);

    void Add(VAddr base_address, size_t size, const T& value) {
        const VAddr end_address = base_address + size;
        IntervalType interval{base_address, end_address};
        m_ranges_map.add({interval, value});
    }

    void Subtract(VAddr base_address, size_t size) {
        const VAddr end_address = base_address + size;
        IntervalType interval{base_address, end_address};
        m_ranges_map -= interval;
    }

    void Clear() {
        m_ranges_map.clear();
    }

    bool Contains(VAddr base_address, size_t size) const {
        const VAddr end_address = base_address + size;
        IntervalType interval{base_address, end_address};
        return boost::icl::contains(m_ranges_map, interval);
    }

    bool Intersects(VAddr base_address, size_t size) const {
        const VAddr end_address = base_address + size;
        IntervalType interval{base_address, end_address};
        return boost::icl::intersects(m_ranges_map, interval);
    }

    template <typename Func>
    void ForEach(Func&& func) const {
        if (m_ranges_map.empty()) {
            return;
        }

        for (const auto& [interval, value] : m_ranges_map) {
            const VAddr inter_addr_end = interval.upper();
            const VAddr inter_addr = interval.lower();
            func(inter_addr, inter_addr_end, value);
        }
    }

    template <typename Func>
    void ForEachInRange(VAddr base_addr, size_t size, Func&& func) const {
        if (m_ranges_map.empty()) {
            return;
        }
        const VAddr start_address = base_addr;
        const VAddr end_address = start_address + size;
        // fifa-hybrid: lower_bound/upper_bound with the whole query interval skipped overlapping
        // intervals (boost 1.88, 64-bit domain): a GPU-written range starting before a page run,
        // or the second range inside it, was never written back and then dropped (FIFA's type
        // table, DMA-copied at match start, came back as garbage and a culling shader looped for
        // a second per frame). Start from the interval around the first byte and walk while the
        // intervals begin before the end (verified against a brute-force reference).
        auto it = m_ranges_map.lower_bound(IntervalType{start_address, start_address + 1});
        if (it != m_ranges_map.begin()) {
            const auto prev = std::prev(it);
            if (prev->first.upper() > start_address) {
                it = prev;
            }
        }
        for (; it != m_ranges_map.end() && it->first.lower() < end_address; ++it) {
            if (it->first.upper() <= start_address) {
                continue;
            }
            func(std::max<VAddr>(it->first.lower(), start_address),
                 std::min<VAddr>(it->first.upper(), end_address), it->second);
        }
    }

    template <typename Func>
    void ForEachNotInRange(VAddr base_addr, size_t size, Func&& func) const {
        const VAddr end_addr = base_addr + size;
        ForEachInRange(base_addr, size, [&](VAddr range_addr, VAddr range_end, const T&) {
            if (size_t gap_size = range_addr - base_addr; gap_size != 0) {
                func(base_addr, gap_size);
            }
            base_addr = range_end;
        });
        if (base_addr != end_addr) {
            func(base_addr, end_addr - base_addr);
        }
    }

private:
    IntervalMap m_ranges_map;
};

template <typename T>
class SplitRangeMap {
public:
    using IntervalMap = boost::icl::split_interval_map<
        VAddr, T, boost::icl::total_absorber, std::less, boost::icl::inplace_identity,
        boost::icl::inter_section, ICL_INTERVAL_INSTANCE(ICL_INTERVAL_DEFAULT, VAddr, std::less),
        RangeSetsAllocator>;
    using IntervalType = typename IntervalMap::interval_type;

public:
    SplitRangeMap() = default;
    ~SplitRangeMap() = default;

    SplitRangeMap(SplitRangeMap const&) = delete;
    SplitRangeMap& operator=(SplitRangeMap const&) = delete;

    SplitRangeMap(SplitRangeMap&& other);
    SplitRangeMap& operator=(SplitRangeMap&& other);

    void Add(VAddr base_address, size_t size, const T& value) {
        const VAddr end_address = base_address + size;
        IntervalType interval{base_address, end_address};
        m_ranges_map.add({interval, value});
    }

    void Subtract(VAddr base_address, size_t size) {
        const VAddr end_address = base_address + size;
        IntervalType interval{base_address, end_address};
        m_ranges_map -= interval;
    }

    void Clear() {
        m_ranges_map.clear();
    }

    bool Contains(VAddr base_address, size_t size) const {
        const VAddr end_address = base_address + size;
        IntervalType interval{base_address, end_address};
        return boost::icl::contains(m_ranges_map, interval);
    }

    bool Intersects(VAddr base_address, size_t size) const {
        const VAddr end_address = base_address + size;
        IntervalType interval{base_address, end_address};
        return boost::icl::intersects(m_ranges_map, interval);
    }

    template <typename Func>
    void ForEach(Func&& func) const {
        if (m_ranges_map.empty()) {
            return;
        }

        for (const auto& [interval, value] : m_ranges_map) {
            const VAddr inter_addr_end = interval.upper();
            const VAddr inter_addr = interval.lower();
            func(inter_addr, inter_addr_end, value);
        }
    }

    template <typename Func>
    void ForEachInRange(VAddr base_addr, size_t size, Func&& func) const {
        if (m_ranges_map.empty()) {
            return;
        }
        const VAddr start_address = base_addr;
        const VAddr end_address = start_address + size;
        // fifa-hybrid: same iteration as RangeSet::ForEachInRange (see there).
        auto it = m_ranges_map.lower_bound(IntervalType{start_address, start_address + 1});
        if (it != m_ranges_map.begin()) {
            const auto prev = std::prev(it);
            if (prev->first.upper() > start_address) {
                it = prev;
            }
        }
        for (; it != m_ranges_map.end() && it->first.lower() < end_address; ++it) {
            if (it->first.upper() <= start_address) {
                continue;
            }
            func(std::max<VAddr>(it->first.lower(), start_address),
                 std::min<VAddr>(it->first.upper(), end_address), it->second);
        }
    }

    template <typename Func>
    void ForEachNotInRange(VAddr base_addr, size_t size, Func&& func) const {
        const VAddr end_addr = base_addr + size;
        ForEachInRange(base_addr, size, [&](VAddr range_addr, VAddr range_end, const T&) {
            if (size_t gap_size = range_addr - base_addr; gap_size != 0) {
                func(base_addr, gap_size);
            }
            base_addr = range_end;
        });
        if (base_addr != end_addr) {
            func(base_addr, end_addr - base_addr);
        }
    }

private:
    IntervalMap m_ranges_map;
};

} // namespace VideoCore
