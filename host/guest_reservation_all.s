// fifa-hybrid: bbgpu-host-macos26 layout probe. One zero-fill segment, not pinned, placed by the
// linker right after __DATA (so at ~0x100010000 + slide), large enough to reach past the high
// guest range end 0xfc00000000: it covers both guest ranges without -segaddr. Size 0xFB00000000.
.globl _bbgpu_guest_reservation_all
.zerofill __GUESTALL,__rsv,_bbgpu_guest_reservation_all,0xFB00000000,14
