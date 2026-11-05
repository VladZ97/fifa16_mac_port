# fifa_mac_port — FIFA 16 (PS4) on Apple Silicon, rendered natively in Metal

This repository is the GPU side of a hybrid emulator that runs the PlayStation 4 version of
**FIFA 16** (CUSA02138, update 1.07) on an Apple Silicon Mac at 60 fps:

- **shadPS4** (branch `metal-hybrid` of [`shadps4-fifa`](https://github.com/VladZ97/shadps4-fifa))
  is the game's runtime: kernel, system libraries, memory, input, audio. The game's own x86-64
  code runs under Rosetta 2.
- **this repository** (branch `fifa-hybrid`, a fork of Mark Hall's
  [bloodborne_mac_port](https://github.com/Wooloomooloo2/bloodborne_mac_port)) provides the video
  core as `libbbgpu`: shadPS4's GPU command processor and shader recompiler driving a native Metal
  renderer, running in a separate native arm64 process (`bbgpu-host`) that owns the window.
- a small bridge in shadPS4 (`src/core/bbgpu_bridge.*`, `bbgpu_split/`) connects the two: shared
  guest memory at identical addresses, submit rings, page-protection relays.

Measured on a MacBook Pro M1 Pro (32 GB, macOS 27): open play locked at 60 fps with a 720p
internal resolution presented at the display's resolution, 16.7 ms frames, menus and replays
working, the FIFA Infinity Patch (PC) installed on top. Nothing from the game is in this repo.
You need your own legally dumped copy.

The companion repositories:

| repo | role |
|---|---|
| [`shadps4-fifa`](https://github.com/VladZ97/shadps4-fifa) | emulator fork: `metal-hybrid` (this renderer) and `rain-probe` (Vulkan-only side build) |
| [`fifa16-shadps4-tools`](https://github.com/VladZ97/fifa16-shadps4-tools) | runner script, GPU host app wrapper, launcher sources, test tooling, notes |
| [`fifa16-mods`](https://github.com/VladZ97/fifa16-mods) | `fifamod`: mod manager, PC-to-PS4 asset fixers, per-competition scoreboards |
| [`pes2fifa`](https://github.com/VladZ97/pes2fifa) | PES 2021 stadium converter |

---

## Quickstart (bundled release, no building)

Everything except the game is in one download on the
[Releases](https://github.com/VladZ97/fifa_mac_port/releases) page.

1. Put your dumped game in one folder, as decrypted folders:
   `<folder>/CUSA02138` (base game) and `<folder>/CUSA02138-UPDATE` (update 1.07).
2. Download `FIFA16-macOS-<version>.zip`, unzip it, and in Terminal:

   ```bash
   bash install.sh --game "<folder>"
   ```

   It checks for Apple Silicon, macOS 15+, Rosetta 2 (and installs it if missing) and
   python3 (from the Command Line Tools, `xcode-select --install`), copies the emulator, the
   Metal GPU library and host, the runner and the mod manager into place, points the run script
   and the emulator config at your game, clears the download quarantine, signs the binaries
   ad hoc and installs the `fifamod` command.
3. Open `~/Applications/FIFA 16.app` (right-click, Open the first time) and press **Play**.
   Or, in Terminal: `bash ~/fifa_metal/run.sh play`.

The first start takes about a minute while Rosetta translates the emulator, and the first few
sessions stutter while shaders are compiled into the cache; after that it holds 60 fps. Logs are
in `~/fifa_metal/runs/<timestamp>/emulator.log`. To add the FIFA Infinity Patch and mods, see
section 8. The rest of this page is the build-from-source route and the reference.

## 1. Requirements

- Apple Silicon Mac. Tested on M1 Pro 32 GB; 16 GB should work, less is untested.
- macOS 15 or newer (binaries target 15.0; developed on macOS 27). Rosetta 2 installed:
  `softwareupdate --install-rosetta`.
- Xcode Command Line Tools (`xcode-select --install`) and Homebrew with
  `cmake ninja glslang pkgconf python3`.
- About 15 GB free for the toolchains and builds, plus the game (around 25 GB) and, if you want
  it, the FIFA Infinity Patch (40 GB, on an external SSD is fine).
- The game as decrypted folders, the way a PS4 dump tool produces them, in one directory:

```
<game dir>/
  CUSA02138/            base game  (eboot.bin, sce_sys/, data_*.big, ...)
  CUSA02138-UPDATE/     update 1.07 (same layout; shadPS4 overlays it over the base)
  CUSA02138-mods/       root .ini overrides, created later by fifamod (may be empty)
```

- A controller. A DualShock 4 over Bluetooth works out of the box through SDL.

All scripts assume a fixed layout under your home folder. Keep it, or edit the paths at the top
of each script:

```
~/fifa_metal/bloodborne_mac_port/   this repo (branch fifa-hybrid)
~/fifa_metal/shadps4-metal/         shadps4-fifa (branch metal-hybrid)
~/fifa_metal/run.sh, bbport.ini, host_app/, launcher/, *.py   from fifa16-shadps4-tools/metal
~/FIFA16Mods/                        fifa16-mods (optional, for the patch and mods)
~/Library/Application Support/shadPS4QtLauncher/versions/Custom-metal/   installed emulator
```

## 2. Get the sources

```bash
mkdir -p ~/fifa_metal && cd ~/fifa_metal
git clone -b fifa-hybrid https://github.com/VladZ97/fifa_mac_port.git bloodborne_mac_port
git clone -b metal-hybrid --recurse-submodules --shallow-submodules \
    https://github.com/VladZ97/shadps4-fifa.git shadps4-metal
git clone https://github.com/VladZ97/fifa16-shadps4-tools.git /tmp/tools
cp -R /tmp/tools/metal/. ~/fifa_metal/          # run.sh, bbport.ini, host_app/, launcher/, fixers
cp /tmp/tools/home/*.sh ~/                       # fifa_run.sh and friends (optional helpers)
```

## 3. Build the GPU library (this repo)

Two builds of the same library: an x86-64 one that the game process loads (for the in-process
fallback) and the native arm64 one used by the GPU host.

```bash
cd ~/fifa_metal/bloodborne_mac_port
bash scripts/macos/bootstrap_deps.sh                       # x86-64 deps (SDL3, SPIRV-Cross, ...) -> deps/x86_64
bash build_macos.sh --gpu-only                             # -> out/gpu/libbbgpu.dylib
BB_DEPS_ARCH=arm64 bash scripts/macos/bootstrap_deps.sh    # arm64 deps -> deps/arm64
bash scripts/macos/build_gpu_arm64.sh                      # -> out/gpu-arm64/libbbgpu.dylib
```

The dependency bootstrap downloads and builds several libraries and takes a while. Rebuilding the
library after a change is `cmake --build out/gpu` and `ninja -C out/gpu-arm64 bbgpu`.

## 4. Build the GPU host process and wrap it as an app

```bash
bash ~/fifa_metal/build_host.sh     # -> out/bbgpu-host and out/bbgpu-host-macos27 (arm64)
```

The host runs inside an app bundle so macOS treats it as a game (Game Mode, direct presentation,
full screen below the notch). Create it once from the files in the tools repo, then refresh after
every host rebuild:

```bash
A="$HOME/fifa_metal/host_app/FIFA 16 GPU.app"
mkdir -p "$A/Contents/MacOS"
cp ~/fifa_metal/host_app/Info.plist "$A/Contents/"      # from fifa16-shadps4-tools/metal/host_app
bash ~/fifa_metal/host_app/refresh.sh                   # copies the two host binaries, ad-hoc signs
```

## 5. Build the emulator (shadps4-fifa, branch metal-hybrid)

```bash
cd ~/fifa_metal/shadps4-metal
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_ARCHITECTURES=x86_64 \
      -DENABLE_BBGPU=ON -DBBGPU_ROOT="$HOME/fifa_metal/bloodborne_mac_port"
cmake --build build --parallel 8          # ~20 min the first time; build/shadps4
```

`ENABLE_BBGPU` links the bridge; `BBGPU_ROOT` points at this repository so the build finds
`libbbgpu` and its headers. The result is an x86-64 binary (it must be, the game is x86-64).

Install it as a version of the shadPS4 Qt launcher (the run script and the launcher app look for
it there):

```bash
bash ~/fifa_metal/install.sh
# -> ~/Library/Application Support/shadPS4QtLauncher/versions/Custom-metal/{shadps4, libvulkan*.dylib, kosmickrisp_mesa_icd.json}
```

Then create the emulator's user folder next to it:

```bash
U="$HOME/Library/Application Support/shadPS4QtLauncher/versions/Custom-metal/user"
mkdir -p "$U/download" "$U/cache"
```

Start the emulator once with your game to let it write a default `config.json` in that folder
(`run.sh play` below does this), then set these keys, which are what the 60 fps results use:

| section | key | value | why |
|---|---|---|---|
| GPU | `present_mode` | `Immediate` | lowest latency; the host paces frames itself |
| GPU | `fast_shader_math` | `true` | 15% GPU time in the kit screen, no visible change |
| GPU | `internal_screen_width/height` | `1280` / `720` | internal render size; the host presents at display size |
| GPU | `direct_memory_access_enabled` | `true` | needed for the shared-memory GPU process |
| GPU | `copy_gpu_buffers` | `false` | measured: no gain, more stalls |
| GPU | `fsr_enabled` | `false` | upscaling is the host's job (`bbport.ini`) |
| Audio | `sdl_mic_device` / `openal_mic_device` | `None` | an open microphone cost the game 2 fps |
| Log | `filter` | `*:error` | play mode; `run.sh start` clears it for full logs |

Add your game folder under `General.install_dirs` (the Qt launcher does this too).

## 6. Point the run script at your game and play

`~/fifa_metal/run.sh` is the single entry point. Edit `GAME=` near the top to your
`<game dir>/CUSA02138/eboot.bin` (or export `SIDE_GAME=<that path>`), then:

```bash
bash ~/fifa_metal/run.sh play      # launch for playing: log reduced to errors
bash ~/fifa_metal/run.sh start     # launch with full logs, screenshots and stats (diagnostics)
bash ~/fifa_metal/run.sh status    # running? fps trend, GPU load, last errors
bash ~/fifa_metal/run.sh log       # tail the current run's emulator.log
bash ~/fifa_metal/run.sh stop
```

Each run writes `~/fifa_metal/runs/<timestamp>/emulator.log`. The first launch of a freshly
built emulator is slow to appear: Rosetta translates the 68 MB binary once, then caches it. The
first few sessions also compile shaders into the pipeline archive
(`Custom-metal/user/cache/CUSA02138/metal/pipelines.binaryarchive`); after that, menus and
matches load without stutter.

Knobs `run.sh` passes to the GPU side (set in the environment to override):

| variable | default | meaning |
|---|---|---|
| `SHAD_BBGPU` / `SHAD_BBGPU_SPLIT` | `1` / `1` | use libbbgpu, in the separate host process |
| `BB_GPU_HOST` | `host_app/FIFA 16 GPU.app/Contents/MacOS/bbgpu-host` | host binary |
| `BB_CONFIG` | `~/fifa_metal/bbport.ini` | renderer settings (`renderer=metal`, `upscaler=off`, `show_fps=0`) |
| `BB_FULLSCREEN` | `1` | `0` for a window |
| `BB_PRESENT_MODE` | unset | `Immediate` = display sync off, 5 ms lower latency, may tear |
| `BB_IMMEDIATE_FLIPS` / `BB_MIDFRAME_FLUSH_MS` | `1` / `3` | pacing: present as soon as a frame is done, flush mid-frame every 3 ms |
| `BB_READBACKS` / `BB_READBACK_LINEAR` | `2` / `1` | precise GPU-to-CPU readbacks (the game reads composited kits and faces back) |
| `BB_GPU_WATCHDOG_MS` | `2000` | kill a hung GPU submission instead of hanging the Mac |
| `BB_FRAME_STATS` | unset | `1` prints `[FSTAT]` lines (fps, frame-time percentiles, GPU ms) every 5 s |
| `MTL_HUD_ENABLED` | unset | `1` shows Apple's Metal performance HUD |
| `SHAD_MAPPING_BASE` / `BB_ARENA_LAYOUT` | `400000000` / `runs` | where guest memory lives; do not change |

## 7. The launcher app (Releases)

[Releases](https://github.com/VladZ97/fifa_mac_port/releases) carry the bundle above and, on its
own, **FIFA 16.app**, a small
SwiftUI launcher: an installation checklist (emulator, game, update, GPU host, config, shader
cache), Play / Stop, the log, and the few settings that matter (window or full screen, low-latency
presentation, frame stats, Metal HUD, play or diagnostics mode). It runs the same
`bash ~/fifa_metal/run.sh play` you would type, so it needs the layout above. Unzip it anywhere,
right-click, Open the first time (it is ad-hoc signed). Sources are in
`fifa16-shadps4-tools/metal/launcher` (`bash build.sh` rebuilds it, `--test` runs its model tests).

## 8. Optional: the FIFA Infinity Patch and mods

The PC patches work on the PS4 game because it is the same engine and asset formats, with a few
PS4-specific differences that the mod manager fixes automatically. With
[`fifa16-mods`](https://github.com/VladZ97/fifa16-mods):

```bash
git clone https://github.com/VladZ97/fifa16-mods.git ~/FIFA16Mods
ln -s ~/FIFA16Mods/fifamod.py /opt/homebrew/bin/fifamod && chmod +x ~/FIFA16Mods/fifamod.py
fifamod init --base "<FIP Manual Install folder>" --game "<game dir>/CUSA02138" --mods-dir "<SSD>/FIFA16Mods/mods"
fifamod link "$HOME/Library/Application Support/shadPS4QtLauncher/versions/Custom-metal/user/download/CUSA02138"
fifamod install ~/Downloads/some-scoreboard.rar      # each mod in its own folder, fixers applied
fifamod scoreboard 53 laliga                           # per-competition scoreboards (see below)
fifamod list / enable / disable / remove / conflicts / status
```

What the fixers handle, learned the hard way: facepack hair meshes use a PC vertex layout that
explodes across the pitch on PS4 (converted on install), compressed DDS UI crests render blank
(converted), empty placeholder `.rx3` files crash the PS4 build (dropped), a loose `fonts.txt`
replaces the packed one (merged with the stock list), high-poly ball meshes crash the Vulkan build
(the Metal hybrid plays them). The emulator fork also serves scoreboards per competition: the
game names the competition in its match-dressing requests, and `fifamod scoreboard` builds the
folders the emulator redirects to, with a rule derived from the game's own database (a league's
scoreboard only when both clubs belong to that league; a tournament's whenever it is played).

## 9. Known issues

- Loading a match very rarely deadlocks (every game thread waits on a game mutex); stop and
  relaunch. Being investigated.
- Stadium textures that the game compresses on the GPU use a path Metal cannot write; the renderer
  emulates it ("block shadow" images). Occasional flat brown stadium shells in replays come from
  this path.
- Only offline modes. No PSN, no online features.
- The Vulkan side build (`rain-probe`, via KosmicKrisp) is kept for comparison; it needs the ball
  meshes converted to triangle strips and tops out around 35 fps in open play.

## 10. Troubleshooting

- `run.sh status` and `run.sh log` first. Errors are in `~/fifa_metal/runs/<timestamp>/emulator.log`.
- `Assertion Failed` in `controller.cpp` at startup: a virtual gamepad from a remote-desktop tool;
  fixed in the fork, update the emulator.
- `Unhandled access violation at 0xcdcdcdcdcdcdcdcd` when loading a second career: module reload;
  fixed in the fork.
- `stack_chk_fail` on the render thread at the first ball touch: a high-poly ball mesh on the
  Vulkan build; use the Metal build or `fifamod`'s converted balls.
- The window shows a frozen frame while the log keeps printing `[FSTAT]` lines: the window is
  occluded (another app in front, display asleep); frames resume when it is visible.
- No controller input in the GPU host's full-screen window: the pad is read by the emulator
  process; make sure the game window has focus (click it once).

## Credits and license

- [shadPS4](https://github.com/shadps4-emu/shadPS4) and its contributors, GPL-2.0-or-later.
- [bloodborne_mac_port](https://github.com/Wooloomooloo2/bloodborne_mac_port) by Mark Hall,
  GPL-2.0-or-later: the native Metal renderer, the GPU host process design and the launcher
  pattern this project builds on. This repository keeps that license (see `LICENSE`); the FIFA
  changes are on branch `fifa-hybrid`, upstream history is on `master`.
- The FIFA Infinity Patch team and the scoreboard, face and stadium mod authors whose work the
  mod tooling installs.
- FIFA 16 is a trademark of Electronic Arts. No game files are distributed here.
