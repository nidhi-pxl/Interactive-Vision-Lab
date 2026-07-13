# Demo 1 — Hand Tracking

## Goal

Create a reusable HandTracker class capable of

- opening webcam
- detecting hands
- returning landmarks
- drawing landmarks
- calculating FPS

## Theory

Module
→ A Python file that contains reusable code.

Import
→ Bringing code from another module into the current file.

Class
→ A blueprint for creating objects.

Object
→ An instance created from a class.

Method
→ A function that belongs to a class.

OpenCV (cv2)
→ Handles webcam input, image processing, and drawing. 
→ Colours are stored as BGR (so convert to RBG so MediaPipe can read)
→ Stores frames as (480,640,3)= (height, width, channels) so 640 x 480 is resolution and 3 is BlueGreenRed

NumPy (np)
→ Efficient arrays for storing and manipulating image data.

MediaPipe (mp)
→ AI library that detects hands, faces, and poses.

math
→ Built-in mathematics functions such as distance calculations.

# Demo 2 — Gesture Engine- Pipeline progress
Camera Frame
      │
      ▼
HandTracker.process_frame()
      │
      ▼
MediaPipe Results
      │
      ▼
Landmark Extraction
      │
      ▼
Finger State Detection
      │
      ▼
Gesture Recognition
      │
      ▼
Application Logic

# Demo 3 – Pinch Detection

## Goal

Create a reusable PinchDetector that measures the distance between the thumb tip and index fingertip to determine whether a pinch gesture is occurring.

## Concepts

- Euclidean distance
- Threshold-based classification
- Landmark geometry
- Interaction state
- Reusable interaction modules

## Future Uses

- Hand Painter
- Particle Trails
- Blooming Flowers
- Fruit Ninja
- AR object manipulation
- UI selection