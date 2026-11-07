/* fifa-hybrid: exercises host/guest_takeover.h outside the GPU host.
 *   takeover_test            take the low guest range over, then use it the way the host does
 *   takeover_test block      plant a 1 MiB mapping in the range first: it must be named and refused
 *   takeover_test walk       only describe what the low and high ranges hold
 * Build: clang -arch arm64 -mmacosx-version-min=15.0 -fobjc-arc -I host host/tests/takeover_test.m \
 *        -framework Metal -framework Foundation -o takeover_test */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdarg.h>
#include <stdio.h>
#include "guest_takeover.h"

#define LOW_MIN UINT64_C(0x400000000)
#define LOW_MAX UINT64_C(0xfa0000000)
#define HIGH_MIN UINT64_C(0x7000000000)
#define HIGH_MAX UINT64_C(0xfc00000000)

static void say(const char *format, ...) {
    va_list args;
    va_start(args, format);
    fputs("test: ", stdout);
    vprintf(format, args);
    fputc('\n', stdout);
    va_end(args);
}

/* What the renderer does with the range: guest memory mirrored at its guest addresses (FIFA maps
 * 0x406200000 + 1.6 GiB and 0x469d00000 for the GPU) is wrapped in no-copy Metal buffers; shaders
 * and blits read and write it in place. Returns the number of failures. */
static int gpu_over_taken_range(id<MTLDevice> device, id<MTLCommandQueue> queue) {
    static const uint64_t spots[] = {UINT64_C(0x406200000), UINT64_C(0x469d00000), UINT64_C(0x800000000), UINT64_C(0xf80000000)};
    enum { Count = 4 };
    const mach_vm_size_t bytes = 32 << 20; /* per mirror */
    const NSUInteger words = (NSUInteger)(bytes / 4);
    int failures = 0;
    id<MTLBuffer> wrapped[Count];
    uint32_t *alias[Count]; /* the same pages, mapped elsewhere (the game's own view, as it were) */
    for (int i = 0; i < Count; ++i) {
        mach_vm_address_t backing = 0;
        mach_port_t entry = MACH_PORT_NULL;
        memory_object_size_t size = bytes;
        kern_return_t kr = mach_vm_allocate(mach_task_self(), &backing, bytes, VM_FLAGS_ANYWHERE);
        if (kr == KERN_SUCCESS) kr = mach_make_memory_entry_64(mach_task_self(), &size, backing, VM_PROT_READ | VM_PROT_WRITE, &entry, MACH_PORT_NULL);
        mach_vm_address_t where = spots[i];
        if (kr == KERN_SUCCESS)
            kr = mach_vm_map(mach_task_self(), &where, bytes, 0, VM_FLAGS_FIXED | VM_FLAGS_OVERWRITE, entry, 0, FALSE,
                             VM_PROT_READ | VM_PROT_WRITE, VM_PROT_READ | VM_PROT_WRITE, VM_INHERIT_NONE);
        if (kr != KERN_SUCCESS) { say("GPU test: mirror at 0x%" PRIx64 " failed: %s", spots[i], mach_error_string(kr)); return failures + 1; }
        alias[i] = (uint32_t *)(uintptr_t)backing;
        for (NSUInteger w = 0; w < words; ++w) alias[i][w] = (uint32_t)(w * 2654435761u + (uint32_t)i);
        wrapped[i] = [device newBufferWithBytesNoCopy:(void *)(uintptr_t)spots[i] length:bytes options:MTLResourceStorageModeShared deallocator:nil];
        if (!wrapped[i]) { say("GPU test: no-copy buffer at 0x%" PRIx64 " was refused", spots[i]); return failures + 1; }
    }
    NSError *error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:@"#include <metal_stdlib>\nusing namespace metal;\n"
        "kernel void mix(device const uint *a [[buffer(0)]], device uint *b [[buffer(1)]], uint i [[thread_position_in_grid]]) { b[i] = a[i] * 3u + b[i] + 1u; }\n"
                                                  options:nil error:&error];
    id<MTLComputePipelineState> pipeline = library ? [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"mix"] error:&error] : nil;
    if (!pipeline) { say("GPU test: shader build failed: %s", error.localizedDescription.UTF8String); return failures + 1; }
    id<MTLBuffer> plain = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:1024 height:1024 mipmapped:NO];
    td.usage = MTLTextureUsageShaderRead;
    id<MTLTexture> texture = [device newTextureWithDescriptor:td];
    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    id<MTLComputeCommandEncoder> compute = [cmd computeCommandEncoder];
    [compute setComputePipelineState:pipeline];
    for (int i = 0; i < Count; ++i) { /* each mirror is read by one dispatch and written by the next */
        [compute setBuffer:wrapped[i] offset:0 atIndex:0];
        [compute setBuffer:wrapped[(i + 1) % Count] offset:0 atIndex:1];
        [compute dispatchThreads:MTLSizeMake(words, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
    [compute endEncoding];
    id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
    [blit copyFromBuffer:wrapped[0] sourceOffset:0 toBuffer:plain destinationOffset:0 size:bytes];
    [blit copyFromBuffer:wrapped[1] sourceOffset:0 sourceBytesPerRow:4096 sourceBytesPerImage:4096 * 1024 sourceSize:MTLSizeMake(1024, 1024, 1)
               toTexture:texture destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0, 0, 0)];
    [blit copyFromTexture:texture sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(1024, 1024, 1)
                 toBuffer:wrapped[3] destinationOffset:16 << 20 destinationBytesPerRow:4096 destinationBytesPerImage:4096 * 1024];
    [blit endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    if (cmd.status != MTLCommandBufferStatusCompleted) { say("GPU test: command buffer failed: %s", cmd.error.localizedDescription.UTF8String); return failures + 1; }
    /* The expected values, computed on the CPU in the same order. */
    uint32_t *expect[Count];
    for (int i = 0; i < Count; ++i) {
        expect[i] = malloc(bytes);
        for (NSUInteger w = 0; w < words; ++w) expect[i][w] = (uint32_t)(w * 2654435761u + (uint32_t)i);
    }
    for (int i = 0; i < Count; ++i) {
        uint32_t *a = expect[i], *b = expect[(i + 1) % Count];
        for (NSUInteger w = 0; w < words; ++w) b[w] = a[w] * 3u + b[w] + 1u;
    }
    memcpy((uint8_t *)expect[3] + (16 << 20), expect[1], 4096 * 1024); /* texture round trip */
    for (int i = 0; i < Count; ++i) {
        const int through_range = memcmp((void *)(uintptr_t)spots[i], expect[i], bytes) == 0;
        const int through_alias = memcmp(alias[i], expect[i], bytes) == 0;
        if (!through_range || !through_alias) {
            say("GPU test: mirror at 0x%" PRIx64 " holds wrong data after the GPU pass (range view %s, other view %s)", spots[i],
                through_range ? "ok" : "WRONG", through_alias ? "ok" : "WRONG");
            ++failures;
        }
    }
    if (memcmp(plain.contents, expect[0], bytes) != 0) { say("GPU test: blit out of the range gave wrong data"); ++failures; }
    say("GPU over the taken range (4 no-copy buffers, compute read/write, buffer and texture blits): %s", failures ? "FAILED" : "ok");
    for (int i = 0; i < Count; ++i) free(expect[i]);
    return failures;
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IOLBF, 0);
    const char *mode = argc > 1 ? argv[1] : "take";
    if (!strcmp(mode, "walk")) {
        int unused = 0;
        say("low range:");
        int n = bb_takeover_walk(LOW_MIN, LOW_MAX, say, 1, &unused);
        say("low range: %d in use, %d unused reservation(s)", n, unused);
        say("high range:");
        n = bb_takeover_walk(HIGH_MIN, HIGH_MAX, say, 1, &unused);
        say("high range: %d in use, %d unused reservation(s)", n, unused);
        return 0;
    }
    void *hint = mmap((void *)LOW_MIN, LOW_MAX - LOW_MIN, PROT_NONE, MAP_PRIVATE | MAP_ANON | MAP_NORESERVE, -1, 0);
    say("hinted mmap of the low range -> %p (%s)", hint, hint == (void *)LOW_MIN ? "honoured" : "not honoured");
    if (hint != MAP_FAILED) munmap(hint, LOW_MAX - LOW_MIN);
    if (!strcmp(mode, "block")) {
        void *blocker = mmap((void *)(LOW_MIN + (UINT64_C(1) << 30)), 1 << 20, PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANON | MAP_FIXED, -1, 0);
        say("planted a 1 MiB mapping at %p", blocker);
        const int status = bbgpu_take_unused_range(LOW_MIN, LOW_MAX, say);
        say("RESULT block: status %d (%s)", status, status ? "refused, as it must be" : "TAKEN OVER: WRONG");
        return status ? 0 : 1;
    }
    int status = bbgpu_take_unused_range(LOW_MIN, LOW_MAX, say);
    say("RESULT take: status %d", status);
    if (status) return 1;
    /* The host maps shared guest memory into the range at fixed addresses and reads/writes it. */
    mach_vm_size_t entry_size = 64 << 20;
    mach_port_t entry = MACH_PORT_NULL;
    mach_vm_address_t backing = 0;
    kern_return_t kr = mach_vm_allocate(mach_task_self(), &backing, entry_size, VM_FLAGS_ANYWHERE);
    memory_object_size_t size = entry_size;
    if (kr == KERN_SUCCESS)
        kr = mach_make_memory_entry_64(mach_task_self(), &size, backing, VM_PROT_READ | VM_PROT_WRITE, &entry, MACH_PORT_NULL);
    int failures = 0;
    for (uint64_t at = LOW_MIN; kr == KERN_SUCCESS && at + entry_size <= LOW_MAX; at += UINT64_C(0x180000000)) {
        mach_vm_address_t where = at;
        kr = mach_vm_map(mach_task_self(), &where, entry_size, 0, VM_FLAGS_FIXED | VM_FLAGS_OVERWRITE, entry, 0, FALSE,
                         VM_PROT_READ | VM_PROT_WRITE, VM_PROT_READ | VM_PROT_WRITE, VM_INHERIT_NONE);
        if (kr != KERN_SUCCESS) { say("mirror map at 0x%" PRIx64 " failed: %s", at, mach_error_string(kr)); ++failures; break; }
        volatile uint64_t *p = (volatile uint64_t *)(uintptr_t)at;
        p[0] = at ^ UINT64_C(0x5a5a);
        p[(entry_size / 8) - 1] = at;
        if (*(volatile uint64_t *)(uintptr_t)backing != (at ^ UINT64_C(0x5a5a))) { say("mirror at 0x%" PRIx64 " does not reach the shared pages", at); ++failures; }
        if (mprotect((void *)(uintptr_t)at, 1 << 20, PROT_NONE) || mprotect((void *)(uintptr_t)at, 1 << 20, PROT_READ | PROT_WRITE)) { say("mprotect at 0x%" PRIx64 " failed: %s", at, strerror(errno)); ++failures; }
    }
    if (kr != KERN_SUCCESS) { say("memory entry setup failed: %s", mach_error_string(kr)); ++failures; }
    say("mirror mappings in the taken range: %s", failures ? "FAILED" : "ok (mapped, written, read back, protection changed)");
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLBuffer> buffer = [device newBufferWithLength:1 << 20 options:MTLResourceStorageModeShared];
        say("Metal after the takeover: device %s, queue %s, buffer %s", device ? device.name.UTF8String : "NONE", queue ? "ok" : "NONE", buffer ? "ok" : "NONE");
        if (!device || !queue || !buffer) ++failures;
        NSString *s = [NSString stringWithFormat:@"%@", [NSDate date]]; /* Foundation still works (shared cache intact) */
        if (s.length == 0) ++failures;
        if (device && queue) failures += gpu_over_taken_range(device, queue);
    }
    say("RESULT use: %s", failures ? "FAILED" : "ok");
    return failures ? 1 : 0;
}
