# Edgerton Airsoft

A smartphone shooting range for airsoft. The phone stands in front of a
target, listens for shots and records the moment of impact in slow motion —
named after [Harold Edgerton](https://en.wikipedia.org/wiki/Harold_Eugene_Edgerton),
whose high-speed photographs made bullets in flight visible.

## What it does

- **Target detection by segmentation.** The app photographs the target board
  and segments it into scoring zones (sub-targets). Two stages: a fast OpenCV
  pass (flood fill in HSV colour space) finds candidate regions, then
  EfficientSAM (ONNX Runtime, on the phone) cuts the region out cleanly. A
  random/variable target zone can be chosen for each round.
- **Weapons by sound.** Each weapon is registered in a profile by recording its
  shot sound (plus FPS, BB weight, hop-up, photo, notes). Audio is monitored
  continuously and calibrated against the room noise (talk, music).
- **Shot-triggered high-speed capture.** A detected shot triggers the camera:
  frame rate is boosted to the device maximum (60/120/240 fps), and a 2-second
  ring buffer of frames and audio means the replay includes the moments
  *before* the shot. The impact is exported as a slow-motion MP4 (MediaCodec +
  MediaMuxer, optional AAC audio), with a PNG-sequence fallback.
- **Hit analysis.** Frame differencing locates the impact and a basic
  trajectory; hits are scored against the segmented zones and drawn as an
  overlay on the live preview.
- **Game modes and training.** *Most points*, *Fastest clear*, *Pistol only*
  and *Drill* (zone × repetitions with voice prompts). Several players,
  per-shot records with time, position and points.
- Guided onboarding: profile → permissions → board → calibration → ready.

## Pipeline

```
Camera (CameraX) ──▶ frame ring buffer (2 s) ─┐
Microphone ───────▶ audio ring buffer ────────┤
                     shot detection ──▶ trigger coordinator (audio + optional motion)
                                              │
                                              ▼
                     slow-motion export ◀── trigger window (1.5 s before / 0.5 s after)
                     hit + trajectory analysis ──▶ scoring against segmented zones
```

## Code map

| Area | File |
|------|------|
| App, UI, game logic | `MainActivity.kt` |
| Shot detection from audio | `audio/ShotDetectionEngine.kt`, `audio/AudioRingBuffer.kt` |
| Camera, fps boost | `camera/CameraSessionController.kt`, `camera/FrameRingBuffer.kt` |
| Trigger | `trigger/TriggerCoordinator.kt`, `motion/MotionAnalyzer.kt` |
| Segmentation | `segmentation/LocalSegmenter.kt`, `segmentation/TwoStageSegmenter.kt`, `sam/SamOnnxHelper.kt` |
| Export | `export/Mp4SlowMotionExporter.kt`, `export/ImageSequenceExporter.kt` |
| Analysis, overlay | `analysis/TrajectoryAnalyzer.kt`, `ui/overlay/DetectionOverlay.kt` |

## Build

The segmentation model is **not** in the repository (about 100 MB). Export
[EfficientSAM](https://github.com/yformer/EfficientSAM) to ONNX and place it in
`app/src/main/assets/efficient_sam.onnx`.
Without it the app falls back to the OpenCV segmentation.

```bash
./gradlew assembleDebug
```

## Status

Prototype, running on the phone. Open: MP4/AAC compatibility across devices,
better audio time-stretching at strong slow motion, an ML shot classifier.

## Tech

Kotlin · Jetpack Compose · CameraX · OpenCV 4.13 · ONNX Runtime (EfficientSAM)
· MediaCodec / MediaMuxer · AudioRecord · Android TextToSpeech
