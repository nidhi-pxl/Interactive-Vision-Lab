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