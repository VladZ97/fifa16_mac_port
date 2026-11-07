// SPDX-FileCopyrightText: Copyright 2026 Mark Hall
/* fifa-hybrid: bbgpu-host-macos26. The executable carries nothing but the guest ranges as
 * pinned zero-fill segments (host/guest_reservation_low.s, host/guest_reservation.s), which the
 * kernel maps at exec, before libmalloc runs; all host code is libbbhost.dylib (host/main.c,
 * host/host_runtime.c). Before macOS 27 a pinned segment in an executable with code breaks
 * dyld's flat-namespace lookups into that executable (the renderer's runtime_* callbacks); with
 * the code in a dylib no lookup targets the executable. The plain bbgpu-host becomes this
 * binary (execv) when its own reservation fails; host/main.c reserve_range takes the exact
 * ranges over from the segments, as in bbgpu-host-macos27.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */
int bbgpu_host_main(int argc, char **argv);

int main(int argc, char **argv) { return bbgpu_host_main(argc, argv); }
