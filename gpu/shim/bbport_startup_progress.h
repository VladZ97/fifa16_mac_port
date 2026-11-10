// SPDX-License-Identifier: GPL-2.0-or-later
// fifa 0.5.1-i: the first launch's progress bar in the game window. Without a patch cache the
// emulator spends most of a minute preparing the game's code before the game draws anything; it
// reports its position in the file named by BB_PROGRESS_FILE ("prepare <0..1000>", see the
// emulator's cpu_patches.cpp). Until the game's first frame this shows a small box with a bar
// over the (still black) window: an AppKit view above the Metal layer, removed for good at the
// first frame, so the presenting path is untouched. A start that finds its caches never sees it.

#pragma once

struct SDL_Window;

namespace BbStartupProgress {

/// Window thread, every event poll (cheap: reads the file five times a second while watching).
void Tick(SDL_Window* window);

} // namespace BbStartupProgress
