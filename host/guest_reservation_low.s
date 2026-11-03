// fifa-hybrid: shadPS4's system areas above the arm64 shared region, [0x400000000, 0xfa0000000),
// as an empty, inaccessible segment reserved at exec (see guest_reservation.s). Pinned with
// -segaddr __GUESTLOW,0x400000000. Size: BBGPU_GUEST_LOW_MAX - BBGPU_GUEST_LOW_MIN.
.globl _bbgpu_guest_reservation_low
.zerofill __GUESTLOW,__rsv,_bbgpu_guest_reservation_low,0xBA0000000,14
