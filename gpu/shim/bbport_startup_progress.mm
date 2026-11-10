// SPDX-License-Identifier: GPL-2.0-or-later
// fifa 0.5.1-i: the first launch's progress bar (bbport_startup_progress.h). ARC.
#import <Cocoa/Cocoa.h>
#include <SDL3/SDL.h>

#include <cstdio>
#include <cstdlib>
#include <string>

#include "bbport_startup_progress.h"
#include "bbport_toggles.h"

namespace BbStartupProgress {

namespace {

enum class State { Unknown, Watching, Off };
State state = State::Unknown;
std::string progress_file;
CFAbsoluteTime next_poll = 0;
NSView* box = nil;
NSProgressIndicator* bar = nil;
NSTextField* detail = nil;
bool finishing = false;

/// 0..1000 from the progress file, -1 while the emulator has not written it.
int ReadPermille() {
    std::FILE* f = std::fopen(progress_file.c_str(), "r");
    if (f == nullptr) {
        return -1;
    }
    unsigned value = 0;
    const int fields = std::fscanf(f, "prepare %u", &value);
    std::fclose(f);
    return fields == 1 ? int(value > 1000 ? 1000 : value) : -1;
}

NSTextField* MakeLabel(NSString* text, NSFont* font, NSColor* color, NSRect frame) {
    NSTextField* label = [NSTextField labelWithString:text];
    label.font = font;
    label.textColor = color;
    label.frame = frame;
    label.alignment = NSTextAlignmentCenter;
    return label;
}

void Show(SDL_Window* window) {
    NSWindow* ns_window = (__bridge NSWindow*)SDL_GetPointerProperty(
        SDL_GetWindowProperties(window), SDL_PROP_WINDOW_COCOA_WINDOW_POINTER, nullptr);
    NSView* content = ns_window.contentView;
    if (content == nil) {
        return;
    }
    const CGFloat width = 440, height = 124;
    const NSRect bounds = content.bounds;
    box = [[NSView alloc] initWithFrame:NSMakeRect((bounds.size.width - width) / 2,
                                                   (bounds.size.height - height) / 2, width,
                                                   height)];
    box.autoresizingMask =
        NSViewMinXMargin | NSViewMaxXMargin | NSViewMinYMargin | NSViewMaxYMargin;
    box.appearance = [NSAppearance appearanceNamed:NSAppearanceNameDarkAqua];
    box.wantsLayer = YES;
    box.layer.backgroundColor = [NSColor colorWithWhite:0.11 alpha:0.94].CGColor;
    box.layer.cornerRadius = 14;

    [box addSubview:MakeLabel(@"Preparing the game for its first launch",
                              [NSFont systemFontOfSize:15 weight:NSFontWeightSemibold],
                              NSColor.whiteColor, NSMakeRect(20, 84, width - 40, 22))];
    bar = [[NSProgressIndicator alloc] initWithFrame:NSMakeRect(28, 56, width - 56, 16)];
    bar.style = NSProgressIndicatorStyleBar;
    bar.indeterminate = NO;
    bar.minValue = 0;
    bar.maxValue = 1000;
    [box addSubview:bar];
    detail = MakeLabel(@"", [NSFont systemFontOfSize:12], NSColor.secondaryLabelColor,
                       NSMakeRect(20, 18, width - 40, 30));
    detail.maximumNumberOfLines = 2;
    [box addSubview:detail];
    [content addSubview:box positioned:NSWindowAbove relativeTo:nil];
}

void Hide() {
    [box removeFromSuperview];
    box = nil;
    bar = nil;
    detail = nil;
}

} // namespace

void Tick(SDL_Window* window) {
    if (state == State::Off) {
        return;
    }
    if (state == State::Unknown) {
        const char* file = std::getenv("BB_PROGRESS_FILE");
        if (file == nullptr || *file == 0 || window == nullptr) {
            state = State::Off;
            return;
        }
        progress_file = file;
        state = State::Watching;
    }
    const CFAbsoluteTime now = CFAbsoluteTimeGetCurrent();
    if (now < next_poll || !NSThread.isMainThread) {
        return;
    }
    next_poll = now + 0.2;
    // The game's first frame ends the start-up: the box never sits over a presented picture.
    if (BbStats::frame_serial.load(std::memory_order_relaxed) > 0) {
        Hide();
        state = State::Off;
        return;
    }
    if (finishing) {
        return;
    }
    const int permille = ReadPermille();
    if (permille < 0) {
        return;
    }
    @autoreleasepool {
        if (box == nil) {
            Show(window);
            if (box == nil) {
                state = State::Off;
                return;
            }
        }
        if (permille >= 1000) {
            // The code is ready; the game still loads for some seconds before its first frame.
            finishing = true;
            bar.indeterminate = YES;
            [bar startAnimation:nil];
            detail.stringValue = @"Starting the game…";
        } else {
            bar.doubleValue = permille;
            detail.stringValue = [NSString
                stringWithFormat:@"%d%%  ·  one time only.\nLater launches "
                                 @"skip this step.",
                                 permille / 10];
        }
    }
}

} // namespace BbStartupProgress
