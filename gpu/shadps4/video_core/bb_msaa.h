// SPDX-FileCopyrightText: Copyright 2026 fifa-hybrid contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <cstdlib>
#include "common/types.h"

namespace VideoCore {

/// 0.5.1-e: BB_MSAA=<n> renders every multisampled guest target with n samples instead of the
/// guest's count. FIFA 16 asks for EQAA (4 coverage samples over 2 colour fragments) and its
/// resolve shader walks 4 coverage samples through the emulated FMASK (0,1,2,3): with 4 real
/// samples that is true 4x multisampling, the PS4's edge quality; with the guest's 2, samples
/// 2 and 3 read as clamped duplicates. Applied wherever a guest sample count becomes a texture
/// or pipeline sample count, so images, attachments and pipelines agree. 0 or unset: off.
inline u32 BbSamples(u32 guest_samples) {
    static const u32 override_samples = [] {
        const char* env = std::getenv("BB_MSAA");
        const int n = env ? std::atoi(env) : 0;
        return n == 2 || n == 4 || n == 8 ? u32(n) : 0u;
    }();
    return guest_samples > 1 && override_samples > guest_samples ? override_samples : guest_samples;
}

} // namespace VideoCore
