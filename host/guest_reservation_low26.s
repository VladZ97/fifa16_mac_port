// fifa-hybrid: bbgpu-host-macos26's low segment: only the first 20 GiB of the low guest range
// [0x400000000, 0xfa0000000). macOS 26's libmalloc must find a 24 GiB window below 64 GiB at
// process start or it aborts the process ("pointer range initial reservation failed"); the
// full 46.5 GiB segment of bbgpu-host-macos27 leaves none. 20 GiB covers the game's system
// managed area above the mapping base (16 GiB) with room to spare; main.c reserve_range takes
// only this part and host_view_apply refuses views beyond it. Pinned at 0x400000000.
.globl _bbgpu_guest_reservation_low26
.zerofill __GUESTLOW,__rsv,_bbgpu_guest_reservation_low26,0x500000000,14
