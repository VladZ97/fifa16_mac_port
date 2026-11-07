/* fifa-hybrid: reserving a guest range the mmap hint cannot get (plain bbgpu-host, before macOS 27).
 *
 * The low guest range [0x400000000, 0xfa0000000) lies inside the dyld shared region's submap,
 * [0x180000000, 0xfc0000000) in arm64 processes. Past the end of the cache the submap holds only
 * an inaccessible, never-touched reservation (or nothing), yet a hinted mmap is never placed
 * inside the region: "cannot reserve the guest range 0x400000000-0xfa0000000 (got 0xfc00000000)"
 * on macOS 26. A fixed mapping can take those addresses: bbgpu-host-macos27 gets the same ones
 * from its exec-time segment (not usable before macOS 27, the pinned segment breaks dyld there).
 *
 * The range is taken over only when everything in it is such unused reservation. Anything else
 * (a mapping at the top level, or one inside the submap that is accessible or was ever touched,
 * like the cache itself) is named and the range is refused: no foreign mapping is overwritten.
 *
 * Used from reserve_range() in main.c, in the branch for a host built without the segment; the
 * -macos27 host never runs it. host/tests/takeover_test.m exercises it on its own. */
#ifndef BBGPU_GUEST_TAKEOVER_H
#define BBGPU_GUEST_TAKEOVER_H

#include <errno.h>
#include <inttypes.h>
#include <libproc.h>
#include <limits.h>
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <stdint.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

typedef void (*BbTakeoverSay)(const char *format, ...);

/* The first region at or after *at, submaps descended into. */
static int bb_takeover_region(mach_vm_address_t *at, mach_vm_size_t *size, natural_t *depth,
                              vm_region_submap_info_data_64_t *info) {
    mach_msg_type_number_t count = VM_REGION_SUBMAP_INFO_COUNT_64;
    *depth = 64;
    *size = 0;
    return mach_vm_region_recurse(mach_task_self(), at, size, depth, (vm_region_recurse_info_t)info, &count) ==
           KERN_SUCCESS;
}

/* Address space that is reserved without holding anything: an entry inside a submap (the shared
 * region), not a submap itself, inaccessible, with no page ever touched. */
static int bb_takeover_unused(natural_t depth, const vm_region_submap_info_data_64_t *info) {
    return depth > 0 && !info->is_submap && info->protection == VM_PROT_NONE && info->pages_resident == 0 &&
           info->pages_swapped_out == 0 && info->pages_dirtied == 0;
}

/* Walks [lo, hi). Returns the number of regions in use; names up to 16 of them through `say`.
 * *unused_out (may be NULL) gets the number of unused reservations found; with `describe`, up to
 * 8 of those are logged too (the layout, for a log from a Mac this was not tested on). */
static int bb_takeover_walk(uint64_t lo, uint64_t hi, BbTakeoverSay say, int describe, int *unused_out) {
    mach_vm_address_t at = lo;
    int in_use = 0, unused = 0;
    for (int guard = 0; guard < 4096; ++guard) {
        mach_vm_size_t size = 0;
        natural_t depth = 0;
        vm_region_submap_info_data_64_t info;
        if (!bb_takeover_region(&at, &size, &depth, &info) || at >= hi || size == 0) break;
        if (bb_takeover_unused(depth, &info)) {
            if (describe && unused < 8)
                say("  unused: 0x%" PRIx64 "-0x%" PRIx64 " (%" PRIu64 " MiB) depth %u prot %d/%d share %d", (uint64_t)at,
                    (uint64_t)(at + size), (uint64_t)(size >> 20), depth, info.protection, info.max_protection,
                    (int)info.share_mode);
            ++unused;
        } else {
            if (in_use < 16) {
                char file[PATH_MAX] = "";
                proc_regionfilename(getpid(), at, file, sizeof file);
                say("  in the way: 0x%" PRIx64 "-0x%" PRIx64 " (%" PRIu64 " MiB) depth %u submap %d tag %u prot %d/%d "
                    "share %d resident %u %s",
                    (uint64_t)at, (uint64_t)(at + size), (uint64_t)(size >> 20), depth, (int)info.is_submap,
                    info.user_tag, info.protection, info.max_protection, (int)info.share_mode, info.pages_resident, file);
            }
            ++in_use;
        }
        at += size;
    }
    if (unused_out) *unused_out = unused;
    return in_use;
}

/* Takes [lo, hi) over with an inaccessible anonymous mapping if nothing in it is in use.
 * Returns 0 on success, an errno otherwise (the reason is logged). */
static int bbgpu_take_unused_range(uint64_t lo, uint64_t hi, BbTakeoverSay say) {
    const uint64_t span = hi - lo;
    int unused = 0;
    const int in_use = bb_takeover_walk(lo, hi, say, 1, &unused);
    if (in_use) {
        say("guest range 0x%" PRIx64 "-0x%" PRIx64 ": %d mapping(s) in use, not taken over", lo, hi, in_use);
        return EBUSY;
    }
    void *fixed = mmap((void *)(uintptr_t)lo, span, PROT_NONE, MAP_PRIVATE | MAP_ANON | MAP_NORESERVE | MAP_FIXED, -1, 0);
    if (fixed != (void *)(uintptr_t)lo) {
        const int mmap_errno = errno;
        say("guest range 0x%" PRIx64 "-0x%" PRIx64 ": fixed mmap failed (%s); trying mach_vm_map", lo, hi,
            strerror(mmap_errno));
        mach_vm_address_t at = lo;
        const kern_return_t kr = mach_vm_map(mach_task_self(), &at, span, 0, VM_FLAGS_FIXED | VM_FLAGS_OVERWRITE,
                                             MEMORY_OBJECT_NULL, 0, FALSE, VM_PROT_NONE, VM_PROT_ALL, VM_INHERIT_NONE);
        if (kr != KERN_SUCCESS || at != lo) {
            say("guest range 0x%" PRIx64 "-0x%" PRIx64 ": mach_vm_map failed too (%s)", lo, hi, mach_error_string(kr));
            return mmap_errno ? mmap_errno : ENOMEM;
        }
    }
    /* The range must now be this mapping alone: nothing of the submap left inside it. */
    mach_vm_address_t at = lo;
    for (int guard = 0; guard < 4096 && at < hi; ++guard) {
        mach_vm_size_t size = 0;
        natural_t depth = 0;
        vm_region_submap_info_data_64_t info;
        if (!bb_takeover_region(&at, &size, &depth, &info) || at >= hi || size == 0) break;
        if (depth != 0 || info.is_submap || info.protection != VM_PROT_NONE) {
            say("guest range 0x%" PRIx64 "-0x%" PRIx64 ": after the takeover 0x%" PRIx64 "-0x%" PRIx64
                " is depth %u submap %d prot %d",
                lo, hi, (uint64_t)at, (uint64_t)(at + size), depth, (int)info.is_submap, info.protection);
            return EINVAL;
        }
        at += size;
    }
    say("guest range 0x%" PRIx64 "-0x%" PRIx64 " taken over from unused address space (%d reservation(s) replaced)", lo,
        hi, unused);
    return 0;
}

#endif
