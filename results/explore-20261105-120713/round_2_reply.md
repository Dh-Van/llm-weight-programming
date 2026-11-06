None of last round's files ran. I wrapped all three in one `{"files": [...]}` object, and the runner wanted `layers` at the top level of each file. Below are the same three feature files, each as its own JSON block, with no changes.

1. **R2F1 – stroke direction by zone (36 neurons):** four 5×5 line detectors (horizontal, vertical, \, /), averaged over a 3×3 grid of zones.
2. **R2F2 – enclosed holes (8 neurons):** fires when there is ink above, below, left and right of an empty pixel. It is checked in the four quadrants at a strict threshold and a lenient one.
3. **R2F3 – stroke endpoints and ink amount (32 neurons):** endpoint count and ink density in each zone of a 4×4 grid.

My predictions:
- Hole neurons should fire for 0, 6, 8, 9 and closed 4s, and stay silent for 1, 2, 3, 5 and 7.
- Vertical-centre and endpoint neurons should pick out 1s.
- Top-horizontal neurons should pick out 7s, 5s and 2s, and the bottom-horizontal neurons 2s.

```json
{"name":"strokes3x3","author":"Claude Opus 5.5","notes":"4 oriented zero-sum 5x5 stroke detectors (horiz, vert, backslash, slash), bias -0.5, relu, avgpool9 -> 3x3 zones (rows/cols 0-8, 9-17, 18-26)",
"layers":[
{"type":"conv2d","padding":2,"weights":[[[[-0.3,-0.3,-0.3,-0.3,-0.3],[0.2,0.2,0.2,0.2,0.2],[0.2,0.2,0.2,0.2,0.2],[0.2,0.2,0.2,0.2,0.2],[-0.3,-0.3,-0.3,-0.3,-0.3]]],[[[-0.3,0.2,0.2,0.2,-0.3],[-0.3,0.2,0.2,0.2,-0.3],[-0.3,0.2,0.2,0.2,-0.3],[-0.3,0.2,0.2,0.2,-0.3],[-0.3,0.2,0.2,0.2,-0.3]]],[[[0.15,0.15,0.0,-0.325,-0.325],[0.15,0.15,0.15,0.0,-0.325],[0.0,0.15,0.15,0.15,0.0],[-0.325,0.0,0.15,0.15,0.15],[-0.325,-0.325,0.0,0.15,0.15]]],[[[-0.325,-0.325,0.0,0.15,0.15],[-0.325,0.0,0.15,0.15,0.15],[0.0,0.15,0.15,0.15,0.0],[0.15,0.15,0.15,0.0,-0.325],[0.15,0.15,0.0,-0.325,-0.325]]]],
"bias":[-0.5,-0.5,-0.5,-0.5],"names":["horiz","vert","diag_backslash","diag_slash"]},
{"type":"relu"},{"type":"avgpool","size":9},{"type":"flatten"}],
"neurons":["horiz TL","horiz TM","horiz TR","horiz ML","horiz C","horiz MR","horiz BL","horiz BM","horiz BR","vert TL","vert TM","vert TR","vert ML","vert C","vert MR","vert BL","vert BM","vert BR","bslash TL","bslash TM","bslash TR","bslash ML","bslash C","bslash MR","bslash BL","bslash BM","bslash BR","slash TL","slash TM","slash TR","slash ML","slash C","slash MR","slash BL","slash BM","slash BR"]}
```

```json
{"name":"holes","author":"Claude Opus 5.5","notes":"14x14: per arm clipped ink min(arm,1)=relu(arm)-relu(arm-1) for up/down/left/right arms of length 4; hole = relu(sum - 2*center - thr); thr 3.4 strict, 3.0 lenient; maxpool7 quadrants",
"layers":[
{"type":"avgpool","size":2},
{"type":"conv2d","padding":4,"weights":[[[[0,0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0]]],[[[0,0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0]]],[[[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0,0]]],[[[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0,0]]],[[[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[1,1,1,1,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0]]],[[[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[1,1,1,1,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0]]],[[[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,1,1,1,1],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0]]],[[[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,1,1,1,1],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0]]],[[[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0]]]],
"bias":[0,-1,0,-1,0,-1,0,-1,0],"names":["up","up_over1","down","down_over1","left","left_over1","right","right_over1","center"]},
{"type":"relu"},
{"type":"conv2d","padding":0,"weights":[[[[1]],[[-1]],[[1]],[[-1]],[[1]],[[-1]],[[1]],[[-1]],[[-2]]],[[[1]],[[-1]],[[1]],[[-1]],[[1]],[[-1]],[[1]],[[-1]],[[-2]]]],"bias":[-3.4,-3.0],"names":["hole_strict","hole_lenient"]},
{"type":"relu"},{"type":"maxpool","size":7},{"type":"flatten"}],
"neurons":["strict hole top-left","strict hole top-right","strict hole bottom-left","strict hole bottom-right","lenient hole top-left","lenient hole top-right","lenient hole bottom-left","lenient hole bottom-right"]}
```

```json
{"name":"ink_endpoints4x4","author":"Claude Opus 5.5","notes":"endpoint = relu(center3x3 - 1.5*ring5x5 - 0.5); ink = pixel; avgpool7 -> 4x4 zones",
"layers":[
{"type":"conv2d","padding":2,"weights":[[[[-1.5,-1.5,-1.5,-1.5,-1.5],[-1.5,1.0,1.0,1.0,-1.5],[-1.5,1.0,1.0,1.0,-1.5],[-1.5,1.0,1.0,1.0,-1.5],[-1.5,-1.5,-1.5,-1.5,-1.5]]],[[[0,0,0,0,0],[0,0,0,0,0],[0,0,1,0,0],[0,0,0,0,0],[0,0,0,0,0]]]],"bias":[-0.5,0],"names":["endpoint","ink"]},
{"type":"relu"},{"type":"avgpool","size":7},{"type":"flatten"}],
"neurons":["endpoint r0c0","endpoint r0c1","endpoint r0c2","endpoint r0c3","endpoint r1c0","endpoint r1c1","endpoint r1c2","endpoint r1c3","endpoint r2c0","endpoint r2c1","endpoint r2c2","endpoint r2c3","endpoint r3c0","endpoint r3c1","endpoint r3c2","endpoint r3c3","ink r0c0","ink r0c1","ink r0c2","ink r0c3","ink r1c0","ink r1c1","ink r1c2","ink r1c3","ink r2c0","ink r2c1","ink r2c2","ink r2c3","ink r3c0","ink r3c1","ink r3c2","ink r3c3"]}
```