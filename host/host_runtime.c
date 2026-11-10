// SPDX-FileCopyrightText: Copyright 2026 Mark Hall
/* bbgpu-host: the runtime_* functions the GPU library (gpu/, built arm64) imports from bb-probe
 * in a single process, provided here from the mirrored guest view table (M3, design "M3 plan").
 * Guest memory is mapped at the runtime's addresses and always read/write here: the game's page
 * protection only matters in bb-probe (design 3.4; the protection relay is M4).
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */
#define _DARWIN_C_SOURCE
#include "host.h"
#include "gpu/bbgpu.h"
#include <errno.h>
#include <inttypes.h>
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <pthread.h>
#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <signal.h>
#include <time.h>
#include <unistd.h>

/* ---- view table: guest views at the runtime's addresses (writer: the ring thread) ---------- */
typedef struct { uint64_t start, end, phys; uint32_t prot; } View;
static View *views;
static size_t view_count, view_capacity;
static uint64_t events_applied, failures, view_bytes;
static pthread_rwlock_t table_lock = PTHREAD_RWLOCK_INITIALIZER;
static mach_port_t entry = MACH_PORT_NULL;
static uint64_t entry_size;

typedef void (*GpuRange)(uintptr_t address, uint64_t size);
static GpuRange hook_map, hook_unmap, hook_invalidate;

void host_runtime_init(mach_port_t guest_entry, uint64_t guest_entry_size) {
    entry = guest_entry;
    entry_size = guest_entry_size;
}

static size_t view_index(uint64_t a) { /* first view with end > a */
    size_t lo = 0, hi = view_count;
    while (lo < hi) { size_t mid = (lo + hi) / 2; if (views[mid].end <= a) lo = mid + 1; else hi = mid; }
    return lo;
}
static void view_insert(size_t at, View v) {
    if (view_count == view_capacity) {
        view_capacity = view_capacity ? view_capacity * 2 : 256;
        views = realloc(views, view_capacity * sizeof *views);
        if (!views) { say("out of memory"); _exit(1); }
    }
    memmove(views + at + 1, views + at, (view_count - at) * sizeof *views);
    views[at] = v;
    ++view_count;
    view_bytes += v.end - v.start;
}
static void split_at(uint64_t a) {
    size_t i = view_index(a);
    if (i == view_count || views[i].start >= a) return;
    View right = views[i];
    right.phys += a - right.start;
    right.start = a;
    views[i].end = a;
    view_bytes -= right.end - right.start; /* view_insert adds it back */
    view_insert(i + 1, right);
}
static void drop(uint64_t start, uint64_t end) {
    split_at(start);
    split_at(end);
    size_t i = view_index(start), n = 0;
    while (i + n < view_count && views[i + n].start < end) view_bytes -= views[i + n].end - views[i + n].start, ++n;
    memmove(views + i, views + i + n, (view_count - i - n) * sizeof *views);
    view_count -= n;
}
static int covered_locked(uint64_t start, uint64_t end) {
    uint64_t at = start;
    for (size_t i = view_index(start); at < end; ++i) {
        if (i == view_count || views[i].start > at) return 0;
        at = views[i].end;
    }
    return 1;
}
int host_view_covered(uint64_t start, uint64_t end) {
    pthread_rwlock_rdlock(&table_lock);
    const int result = covered_locked(start, end);
    pthread_rwlock_unlock(&table_lock);
    return result;
}
void host_view_stats(uint64_t *count, uint64_t *bytes, uint64_t *applied, uint64_t *failed) {
    pthread_rwlock_rdlock(&table_lock);
    *count = view_count;
    *bytes = view_bytes;
    *applied = events_applied;
    *failed = failures;
    pthread_rwlock_unlock(&table_lock);
}

static void fail_view(uint32_t op, uint64_t address, uint64_t size, uint64_t phys, const char *what, int code) {
    if (failures++ < 10)
        say("event %u at 0x%" PRIx64 "+0x%" PRIx64 " (phys 0x%" PRIx64 "): %s failed (%d)", op, address, size, phys,
            what, code);
}

/* Applies one view event; the renderer's hooks (when started) run after the table is updated,
 * outside the lock: they query the table. */
void host_view_apply(uint32_t op, uint32_t prot, uint64_t address, uint64_t size, uint64_t phys) {
    const uint64_t start = address, end = address + size;
    GpuRange hook = NULL;
    pthread_rwlock_wrlock(&table_lock);
    ++events_applied;
    if (!size || (start | size) % BBGPU_GUEST_PAGE || !bbgpu_guest_range_ok(start, end)) {
        fail_view(op, address, size, phys, "range check", EINVAL);
        pthread_rwlock_unlock(&table_lock);
        return;
    }
    if (start < BBGPU_GUEST_LOW_MAX && end > bbgpu_low_reserved_max) {
        /* fifa-hybrid: past the part of the low range this host reserved (macOS 26, see main.c). */
        static int said;
        if (!said++)
            say("view at 0x%" PRIx64 "+0x%" PRIx64 " lies beyond the reserved low range end 0x%" PRIx64
                ": refused (the host on this macOS reserves only the first %" PRIu64 " GiB of the low range)",
                address, size, bbgpu_low_reserved_max, (bbgpu_low_reserved_max - BBGPU_GUEST_LOW_MIN) >> 30);
        fail_view(op, address, size, phys, "beyond the reserved low range", E2BIG);
        pthread_rwlock_unlock(&table_lock);
        return;
    }
    switch (op) {
    case BBGPU_EV_MAP: {
        if (phys % BBGPU_GUEST_PAGE || phys > entry_size || size > entry_size - phys) {
            fail_view(op, address, size, phys, "entry range check", EINVAL);
            break;
        }
        mach_vm_address_t at = start;
        kern_return_t kr = mach_vm_map(mach_task_self(), &at, size, 0, VM_FLAGS_FIXED | VM_FLAGS_OVERWRITE, entry, phys,
                                       FALSE, VM_PROT_READ | VM_PROT_WRITE, VM_PROT_READ | VM_PROT_WRITE, VM_INHERIT_NONE);
        if (kr != KERN_SUCCESS || at != start) { fail_view(op, address, size, phys, "mach_vm_map", kr); break; }
        drop(start, end);
        view_insert(view_index(start), (View){start, end, phys, prot});
        hook = hook_map;
        break;
    }
    case BBGPU_EV_UNMAP:
        /* back to the reservation, as the runtime does */
        if (mmap((void *)start, size, PROT_NONE, MAP_PRIVATE | MAP_ANON | MAP_NORESERVE | MAP_FIXED, -1, 0) == MAP_FAILED) {
            fail_view(op, address, size, phys, "mmap(PROT_NONE)", errno);
            break;
        }
        drop(start, end);
        hook = hook_unmap;
        break;
    case BBGPU_EV_PROTECT:
        split_at(start);
        split_at(end);
        for (size_t i = view_index(start); i < view_count && views[i].start < end; ++i) views[i].prot = prot;
        hook = hook_invalidate; /* as the runtime: a guest mprotect invalidates GPU copies */
        break;
    default:
        fail_view(op, address, size, phys, "unknown op", EINVAL);
    }
    pthread_rwlock_unlock(&table_lock);
    if (hook) hook(start, size);
}

/* ---- runtime_* for the GPU library ------------------------------------------------------- */
uint64_t runtime_disabled_optimizations;
__thread sigjmp_buf *runtime_fault_recover;
static __thread int32_t thread_errno;

int32_t *runtime_errno(void) { return &thread_errno; }

/* BB_TOGGLE_FILE: the renderer's optimization/debug toggles, re-read every 250 ms (as bb-probe
 * does for its own side; the renderer runs here in split mode). */
static void *toggle_watcher(void *path) {
    for (unsigned long long last = ~0ULL;;) {
        FILE *f = fopen(path, "r");
        unsigned long long value = 0;
        if (f) {
            if (fscanf(f, "%llu", &value) != 1) value = 0;
            fclose(f);
        }
        if (value != last) {
            __atomic_store_n(&runtime_disabled_optimizations, (uint64_t)value, __ATOMIC_RELEASE);
            say("disabled optimizations mask=%llu", value);
            last = value;
        }
        struct timespec t = {0, 250000000};
        nanosleep(&t, NULL);
    }
    return NULL;
}

void runtime_memory_set_gpu_hooks(GpuRange map, GpuRange unmap, GpuRange invalidate) {
    const char *toggles = getenv("BB_TOGGLE_FILE");
    static pthread_t watcher;
    if (toggles && !watcher) pthread_create(&watcher, NULL, toggle_watcher, (void *)toggles);
    pthread_rwlock_wrlock(&table_lock);
    hook_map = map;
    hook_unmap = unmap;
    hook_invalidate = invalidate;
    size_t n = view_count;
    View *copy = n ? malloc(n * sizeof *copy) : NULL;
    if (copy) memcpy(copy, views, n * sizeof *copy);
    pthread_rwlock_unlock(&table_lock);
    for (size_t i = 0; map && i < n && copy; ++i) map(copy[i].start, copy[i].end - copy[i].start);
    free(copy);
}

/* Guest-visible writes from the GPU side (fence labels, WriteData, downloads): each piece is one
 * release store, as in the runtime (store_once): the game reads labels without locks, and on
 * arm64 plain stores may become visible out of order. */
static void store_once(unsigned char *dst, const unsigned char *src, uint64_t n) {
    if (n > 64) {
        memcpy(dst, src, n);
        __atomic_thread_fence(__ATOMIC_RELEASE);
        return;
    }
    while (n) {
        const uintptr_t at = (uintptr_t)dst;
        if (n >= 8 && !(at & 7)) { uint64_t v; memcpy(&v, src, 8); __atomic_store_n((uint64_t *)dst, v, __ATOMIC_RELEASE); dst += 8; src += 8; n -= 8; }
        else if (n >= 4 && !(at & 3)) { uint32_t v; memcpy(&v, src, 4); __atomic_store_n((uint32_t *)dst, v, __ATOMIC_RELEASE); dst += 4; src += 4; n -= 4; }
        else if (n >= 2 && !(at & 1)) { uint16_t v; memcpy(&v, src, 2); __atomic_store_n((uint16_t *)dst, v, __ATOMIC_RELEASE); dst += 2; src += 2; n -= 2; }
        else { __atomic_store_n(dst, *src, __ATOMIC_RELEASE); ++dst; ++src; --n; }
    }
}

int runtime_memory_write_backing(uintptr_t address, const void *data, uint64_t size) {
    if (!host_view_covered(address, address + size)) return 0;
    store_once((unsigned char *)address, data, size);
    return 1;
}
/* Game threads live in bb-probe: the spike log's CPU times come through the control block. */
uint64_t runtime_thread_cpu_ns(int which) {
    (void)which;
    return 0;
}
void runtime_game_cpu_since_last(uint64_t *total_ns, uint64_t *busiest_ns, char busiest_name[32]) {
    *total_ns = *busiest_ns = 0;
    busiest_name[0] = 0;
}

void runtime_sampler_flip(uint64_t main_cpu_ns) { (void)main_cpu_ns; } /* game side only */

int runtime_memory_read_backing(uintptr_t address, void *out, uint64_t size) {
    if (!host_view_covered(address, address + size)) return 0;
    memcpy(out, (const void *)address, size);
    return 1;
}
int runtime_memory_is_mapped(uintptr_t address, uint64_t size) {
    return host_view_covered(address, address + size);
}
uint64_t runtime_memory_clamp(uintptr_t address, uint64_t size) {
    pthread_rwlock_rdlock(&table_lock);
    uint64_t length = 0;
    for (size_t i = view_index(address); i < view_count && length < size; ++i) {
        if (views[i].start > address + length) break;
        length = views[i].end - address;
    }
    pthread_rwlock_unlock(&table_lock);
    return length < size ? length : size;
}
int runtime_memory_vma_info(uintptr_t address, int *prot, int *type, uintptr_t *end) {
    pthread_rwlock_rdlock(&table_lock);
    const size_t i = view_index(address);
    const int ok = i < view_count && views[i].start <= address;
    if (ok) {
        if (prot) *prot = (int)views[i].prot;
        if (type) *type = -1; /* the mirror does not carry the direct memory type */
        if (end) *end = views[i].end;
    }
    pthread_rwlock_unlock(&table_lock);
    return ok;
}
int runtime_memory_region(uintptr_t address, uintptr_t *start, uintptr_t *end, int *mapped) {
    pthread_rwlock_rdlock(&table_lock);
    const size_t i = view_index(address);
    const int found = i < view_count;
    if (found && views[i].start <= address) {
        *start = views[i].start; *end = views[i].end; *mapped = 1;
    } else if (found) {
        *start = i ? views[i - 1].end : 0; *end = views[i].start; *mapped = 0;
    }
    pthread_rwlock_unlock(&table_lock);
    return found;
}

/* GPU page tracking (M4): protection only matters where the game runs, so each request goes to
 * bb-probe (protect ring) and returns once applied there, as an mprotect would in one process:
 * a game write after this returns faults, and the fault comes back here (main.c). */
static pthread_mutex_t protect_lock = PTHREAD_MUTEX_INITIALIZER;
static uint32_t protect_seq;
static void game_alive_or_exit(void) {
    if (g_shared->game_pid > 0 && kill(g_shared->game_pid, 0) != 0) {
        say("bb-probe is gone (protect relay); exiting");
        bbgpu_flush_caches(BBGPU_EXIT_FLUSH_MS);
        _exit(0);
    }
}
/* fifa-hybrid: restoring full access (read && write) never has to be waited for: a late
 * unprotect only costs the game an extra fault, while the wait was a cross-process round trip
 * (~0.2 ms) on the GPU command thread for every readback that landed. BB_PROTECT_SYNC_ALL=1
 * restores the wait. The ring is in order, so a later waited call covers these too. */
static int protect_sync_all = -1;
/* 0.5.1-b: the waited round trip is ~0.1-0.2 ms of two thread hand-offs; the default wait spins
 * only 200 iterations (~2 us) before sleeping, so the command thread pays a wake-up on top of
 * every one. BB_PROTECT_SPIN_US spins that long first (the command thread is the cutscene limit). */
static int64_t protect_spin_ns = -1;
/* 0.5.1-b: a batch (memory_tracker.h ForEachUploadRange) posts its protects without waiting and
 * waits once for the last of them before the copies: one round trip per upload instead of one
 * per region. Per thread: the command thread and the draw recording thread may both protect. */
static __thread int protect_batch_depth;
static __thread uint32_t protect_batch_last;
/* 0.5.1-c: direct page protection. With the game's task port (task_for_pid: the emulator binary is
 * signed with com.apple.security.get-task-allow, app/make_perf_app.sh) the host changes the game's
 * page protection itself, one mach_vm_protect per view (a few us, no thread hand-off), instead of
 * the relay through the game's bb:GpuProtect thread (~80 us, two wake-ups, 40-100 times a frame:
 * 0.7-1.5 ms of the command thread per frame in 0.5.1-b, more in cutscenes). The game's per-page
 * kind table (bbport_split.cpp g_page_kind, read by its fault handler) is remapped here and kept
 * current; g_shared->protect_lock orders the two sides. BB_PROTECT_DIRECT=0 keeps the relay. */
static mach_port_t game_task = MACH_PORT_NULL;
static uint8_t *kinds;
static uint64_t kinds_bytes;
static uint64_t direct_failures;
static void shared_protect_lock(void) {
    for (uint32_t spins = 0;; ++spins) {
        uint32_t expected = 0;
        if (__atomic_compare_exchange_n(&g_shared->protect_lock, &expected, 1u, 0, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED)) return;
        bbgpu_cpu_relax();
        if ((spins & 0xffffu) == 0xffffu) game_alive_or_exit(); /* the holder is a game fault handler: microseconds */
    }
}
static void shared_protect_unlock(void) { __atomic_store_n(&g_shared->protect_lock, 0u, __ATOMIC_RELEASE); }
static void map_kinds_once(void) {
    if (kinds || kinds_bytes == UINT64_MAX) return;
    const uint64_t addr = __atomic_load_n(&g_shared->page_kind_addr, __ATOMIC_ACQUIRE);
    const uint64_t bytes = __atomic_load_n(&g_shared->page_kind_bytes, __ATOMIC_ACQUIRE);
    if (!addr || !bytes) return; /* not published yet (an older game shim never does): next call */
    mach_vm_address_t at = 0;
    vm_prot_t cur = 0, max = 0;
    const kern_return_t kr = mach_vm_remap(mach_task_self(), &at, bytes, 0, VM_FLAGS_ANYWHERE, game_task, addr, FALSE,
                                           &cur, &max, VM_INHERIT_NONE);
    if (kr != KERN_SUCCESS) {
        say("direct page protection: cannot map the game's page table (%s); its write faults all relay", mach_error_string(kr));
        kinds_bytes = UINT64_MAX;
        return;
    }
    kinds = (uint8_t *)(uintptr_t)at;
    kinds_bytes = bytes;
    say("direct page protection: the game's page table mapped (%" PRIu64 " KiB)", bytes >> 10);
}
/* task_for_pid answers at once (success, or failure 5 without the entitlement), but a pending
 * authorisation dialog (taskgated: "Developer Tools Access") queues the request behind it for as
 * long as the dialog is open (seen 2026-11-09: the handshake timed out). The call runs on its own
 * thread and the host waits 1.5 s for it; a late answer is dropped. */
static pthread_mutex_t tfp_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t tfp_cond = PTHREAD_COND_INITIALIZER;
static int tfp_done, tfp_wanted = 1, tfp_pid;
static kern_return_t tfp_result;
static mach_port_t tfp_port;
static void *tfp_thread(void *arg) {
    (void)arg;
    mach_port_t port = MACH_PORT_NULL;
    const kern_return_t kr = task_for_pid(mach_task_self(), tfp_pid, &port);
    pthread_mutex_lock(&tfp_lock);
    if (tfp_wanted) { tfp_result = kr; tfp_port = port; }
    else if (kr == KERN_SUCCESS) mach_port_deallocate(mach_task_self(), port); /* too late: relay mode */
    tfp_done = 1;
    pthread_cond_broadcast(&tfp_cond);
    pthread_mutex_unlock(&tfp_lock);
    return NULL;
}
int host_protect_direct_init(int game_pid) {
    const char *e = getenv("BB_PROTECT_DIRECT");
    if (e && e[0] == '0') { say("page protection by relay (BB_PROTECT_DIRECT=0)"); return 0; }
    tfp_pid = game_pid;
    pthread_t thread;
    if (pthread_create(&thread, NULL, tfp_thread, NULL) != 0) { say("page protection by relay: no thread"); return 0; }
    pthread_detach(thread);
    struct timespec deadline;
    clock_gettime(CLOCK_REALTIME, &deadline);
    deadline.tv_nsec += 500000000L; deadline.tv_sec += 1 + deadline.tv_nsec / 1000000000L; deadline.tv_nsec %= 1000000000L;
    pthread_mutex_lock(&tfp_lock);
    while (!tfp_done && pthread_cond_timedwait(&tfp_cond, &tfp_lock, &deadline) == 0) {}
    const int done = tfp_done;
    tfp_wanted = 0;
    const kern_return_t kr = done ? tfp_result : KERN_OPERATION_TIMED_OUT;
    game_task = done && kr == KERN_SUCCESS ? tfp_port : MACH_PORT_NULL;
    pthread_mutex_unlock(&tfp_lock);
    if (kr != KERN_SUCCESS) {
        say("page protection by relay: task_for_pid(%d): %s (the direct path needs the emulator signed with "
            "com.apple.security.get-task-allow)", (int)game_pid,
            done ? mach_error_string(kr) : "no answer in 1.5 s (an authorisation dialog pending?)");
        return 0;
    }
    __atomic_store_n(&g_shared->protect_direct, 1u, __ATOMIC_RELEASE);
    say("direct page protection: the host changes the game's page protection itself (task port)");
    return 1;
}
static void protect_direct(uintptr_t address, uint64_t size, int read, int write) {
    const uint64_t page = 4096; /* the game's (x86_64) page, as its own mprotect */
    const uint64_t begin = address & ~(page - 1), end = (address + size + page - 1) & ~(page - 1);
    map_kinds_once();
    shared_protect_lock();
    pthread_rwlock_rdlock(&table_lock);
    for (size_t i = view_index(begin); i < view_count && views[i].start < end; ++i) {
        const uint64_t s = views[i].start > begin ? views[i].start : begin;
        const uint64_t e = views[i].end < end ? views[i].end : end;
        const uint32_t guest = views[i].prot; /* as the game's runtime_memory_gpu_protect (bbgpu_bridge.cpp) */
        vm_prot_t prot = VM_PROT_NONE;
        if ((read & 1) && (guest & 3)) prot |= VM_PROT_READ;
        if (write && (guest & 2)) prot |= VM_PROT_WRITE;
        if (guest & 4) prot |= VM_PROT_EXECUTE | VM_PROT_READ;
        const kern_return_t kr = mach_vm_protect(game_task, s, e - s, FALSE, prot);
        if (kr != KERN_SUCCESS && direct_failures++ < 5)
            say("direct page protection: mach_vm_protect(0x%" PRIx64 "+0x%" PRIx64 ", %d): %s", s, e - s, (int)prot,
                mach_error_string(kr));
    }
    pthread_rwlock_unlock(&table_lock);
    if (kinds) { /* as the game's NoteProtect: 0x80 | read bits << 1 | writable */
        const uint8_t raw = (uint8_t)(0x80 | ((read & 7) << 1) | (write ? 1 : 0));
        for (uint64_t a = begin; a < end; a += page) {
            if (a >= BBGPU_GUEST_LOW_MIN && a < BBGPU_GUEST_LOW_MAX) {
                const uint64_t slot = (a - BBGPU_GUEST_LOW_MIN) >> 12;
                if (slot < kinds_bytes) kinds[slot] = raw;
            }
        }
    }
    shared_protect_unlock();
}
/* g28 diagnostic (BB_AUDIT): how many bytes of [begin, end) the game can write right now, the first
 * such address, and the kind byte of the page at begin. -1 without the game's task port. */
long long runtime_memory_gpu_debug_writable(uintptr_t begin, uintptr_t end, uintptr_t *first, int *kind) {
    if (first) *first = 0;
    if (kind) *kind = -1;
    if (game_task == MACH_PORT_NULL) return -1;
    map_kinds_once();
    if (kind && kinds && begin >= BBGPU_GUEST_LOW_MIN && begin < BBGPU_GUEST_LOW_MAX) {
        const uint64_t slot = (begin - BBGPU_GUEST_LOW_MIN) >> 12;
        if (slot < kinds_bytes) *kind = kinds[slot];
    }
    const uint64_t b = begin & ~UINT64_C(4095), e = (end + 4095) & ~UINT64_C(4095);
    long long writable = 0;
    mach_vm_address_t a = b;
    while (a < e) {
        mach_vm_size_t sz = 0;
        vm_region_basic_info_data_64_t info;
        mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
        mach_port_t object = MACH_PORT_NULL;
        if (mach_vm_region(game_task, &a, &sz, VM_REGION_BASIC_INFO_64, (vm_region_info_t)&info, &count, &object) !=
            KERN_SUCCESS)
            break;
        if (object != MACH_PORT_NULL) mach_port_deallocate(mach_task_self(), object);
        if (a >= e) break;
        const uint64_t s = a > b ? a : b, t = a + sz < e ? a + sz : e;
        if ((info.protection & VM_PROT_WRITE) && t > s) {
            if (first && !*first) *first = (uintptr_t)s;
            writable += (long long)(t - s);
        }
        a += sz;
    }
    return writable;
}
static void protect_wait(uint32_t seq) {
    for (;;) {
        const uint32_t done = __atomic_load_n(&g_shared->protect_done, __ATOMIC_ACQUIRE);
        if ((int32_t)(done - seq) >= 0) return;
        if (bbgpu_wait_word_spin(&g_shared->protect_done, done, UINT64_C(1000000000),
                                 (uint64_t)(protect_spin_ns > 0 ? protect_spin_ns : 0))) game_alive_or_exit();
    }
}
void runtime_memory_gpu_protect_batch(int begin) {
    if (game_task != MACH_PORT_NULL) return; /* 0.5.1-c: direct protection has nothing to wait for */
    if (begin) {
        ++protect_batch_depth;
        return;
    }
    if (protect_batch_depth > 0 && --protect_batch_depth > 0) return;
    const uint32_t seq = protect_batch_last;
    protect_batch_last = 0;
    if (seq) protect_wait(seq);
}
void runtime_memory_gpu_protect(uintptr_t address, uint64_t size, int read, int write) {
    if (protect_sync_all < 0) {
        const char *e = getenv("BB_PROTECT_SYNC_ALL");
        protect_sync_all = e && e[0] == '1';
        protect_spin_ns = (int64_t)__atomic_load_n(&g_shared->protect_spin_ns, __ATOMIC_ACQUIRE); /* main.c */
    }
    if (game_task != MACH_PORT_NULL) {
        protect_direct(address, size, read, write);
        return;
    }
    pthread_mutex_lock(&protect_lock);
    const uint32_t seq = ++protect_seq;
    const BbGpuProtect record = {{BBGPU_REC_PROTECT, sizeof record}, address, size, (uint32_t)read, (uint32_t)write, seq, 0};
    while (bbgpu_ring_push(g_shared, &g_shared->protect, &record, UINT64_C(1000000000))) game_alive_or_exit();
    pthread_mutex_unlock(&protect_lock);
    if ((read & 1) && write && !protect_sync_all) {
        return;
    }
    if (protect_batch_depth > 0) {
        protect_batch_last = seq; /* the ring is in order: waiting for the last covers all */
        return;
    }
    protect_wait(seq);
}

/* Timebase: the runtime's process clock (CLOCK_MONOTONIC is system-wide) and guest TSC rate. */
uint64_t runtime_process_time_counter(void) {
    return clock_gettime_nsec_np(CLOCK_MONOTONIC) - g_shared->process_start_ns;
}
uint64_t runtime_process_time_us(void) { return runtime_process_time_counter() / 1000; }
uint64_t runtime_tsc_frequency(void) { return g_shared->tsc_hz; }

/* bb-probe-only services: AvPlayer, guest thread bookkeeping, restart. */
int runtime_file_translate(const char *guest, char *out, size_t size) {
    (void)guest; (void)out; (void)size;
    return -1;
}
void runtime_thread_attach_host(const char *name) { (void)name; }
void runtime_thread_set_blocked(const char *what, uint64_t resource) { (void)what; (void)resource; }
void runtime_thread_clear_blocked(void) {}
/* The settings menu's Restart: bb-probe restarts through run.sh (exec); this process ends when
 * it sees the exec (main.c parent_watch) and the new bb-probe starts a new host. */
void runtime_restart(void) {
    say("restart requested: asking bb-probe to restart");
    bbgpu_split_post_restart();
}
