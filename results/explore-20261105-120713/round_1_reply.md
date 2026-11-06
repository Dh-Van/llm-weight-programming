This round tests three kinds of hand-built features. The goal is to see what each neuron actually responds to; this is not a classifier yet.

**R1F1: oriented strokes (36 neurons).** Four 5×5 line detectors (horizontal, vertical, "\" and "/") are pooled over a 3×3 grid of 9-pixel zones. Each kernel sums to zero, so solid blobs and strokes at other angles give about 0, and a clean stroke 3 px wide gives about 3.
- **Horizontal:** strong in the top and bottom zones for 7, 2, 5 and 3, and in the middle zone for 4.
- **Vertical:** central column for 1; left and right zones for 0.
- **Diagonals:** "/" for 7, 9 and slanted 1s; "\" for the upper-right part of 0 and 6.

**R1F2: enclosed holes (8 neurons).** On a 14×14 downsample, the first layer measures ink along four arms (up, down, left, right, each 4 px long) and caps each at 1. The second layer fires only when all four arms hit ink and the centre pixel is empty. There are two versions: strict (threshold 3.4) and lenient (3.0). Each is max-pooled over the four quadrants.
- **Top quadrants:** expect 0, 8, 9 and 4.
- **Bottom quadrants:** expect 0, 6 and 8.
- **Near zero:** 1, 2, 3, 5 and 7.

**R1F3: endpoints and ink (32 neurons).** The endpoint score is the 3×3 centre ink minus 1.5 times the 5×5 ring ink, minus 0.5. It fires at stroke ends and stays at or below 0 in the middle of a line, at junctions and inside blobs. The ink channel is plain pixel density. Both are averaged over a 4×4 grid of 7-pixel zones.
- **Endpoints:** expect top and bottom endpoints for 1, both left endpoints for 3 and few for 0 and 8.
- **Ink:** I expect it to separate 0 from 1 by how much ink sits in the centre.

```json
{"files":[
{"name":"strokes3x3","author":"Claude Opus 5.5","notes":"4 oriented zero-sum 5x5 stroke detectors (horiz, vert, backslash, slash), bias -0.5, relu, avgpool9 -> 3x3 zones (rows/cols 0-8, 9-17, 18-26)",
"layers":[
{"type":"conv2d","padding":2,"weights":[[[[-0.3,-0.3,-0.3,-0.3,-0.3],[0.2,0.2,0.2,0.2,0.2],[0.2,0.2,0.2,0.2,0.2],[0.2,0.2,0.2,0.2,0.2],[-0.3,-0.3,-0.3,-0.3,-0.3]]],[[[-0.3,0.2,0.2,0.2,-0.3],[-0.3,0.2,0.2,0.2,-0.3],[-0.3,0.2,0.2,0.2,-0.3],[-0.3,0.2,0.2,0.2,-0.3],[-0.3,0.2,0.2,0.2,-0.3]]],[[[0.15,0.15,0.0,-0.325,-0.325],[0.15,0.15,0.15,0.0,-0.325],[0.0,0.15,0.15,0.15,0.0],[-0.325,0.0,0.15,0.15,0.15],[-0.325,-0.325,0.0,0.15,0.15]]],[[[-0.325,-0.325,0.0,0.15,0.15],[-0.325,0.0,0.15,0.15,0.15],[0.0,0.15,0.15,0.15,0.0],[0.15,0.15,0.15,0.0,-0.325],[0.15,0.15,0.0,-0.325,-0.325]]]],
"bias":[-0.5,-0.5,-0.5,-0.5],"names":["horiz","vert","diag_backslash","diag_slash"]},
{"type":"relu"},{"type":"avgpool","size":9},{"type":"flatten"}],
"neurons":["horiz TL","horiz TM","horiz TR","horiz ML","horiz C","horiz MR","horiz BL","horiz BM","horiz BR","vert TL","vert TM","vert TR","vert ML","vert C","vert MR","vert BL","vert BM","vert BR","bslash TL","bslash TM","bslash TR","bslash ML","bslash C","bslash MR","bslash BL","bslash BM","bslash BR","slash TL","slash TM","slash TR","slash ML","slash C","slash MR","slash BL","slash BM","slash BR"]},
{"name":"holes","author":"Claude Opus 5.5","notes":"14x14: per arm clipped ink min(arm,1)=relu(arm)-relu(arm-1) for up/down/left/right arms of length 4; hole = relu(sum - 2*center - thr); thr 3.4 strict, 3.0 lenient; maxpool7 quadrants",
"layers":[
{"type":"avgpool","size":2},
{"type":"conv2d","padding":4,"weights":[[[[0,0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0]]],[[[0,0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0]]],[[[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0,0]]],[[[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0,0]]],[[[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[1,1,1,1,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0]]],[[[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[1,1,1,1,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0]]],[[[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,1,1,1,1],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0]]],[[[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,1,1,1,1],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0]]],[[[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0]]]],
"bias":[0,-1,0,-1,0,-1,0,-1,0],"names":["up","up_over1","down","down_over1","left","left_over1","right","right_over1","center"]},
{"type":"relu"},
{"type":"conv2d","padding":0,"weights":[[[[1]],[[-1]],[[1]],[[-1]],[[1]],[[-1]],[[1]],[[-1]],[[-2]]],[[[1]],[[-1]],[[1]],[[-1]],[[1]],[[-1]],[[1]],[[-1]],[[-2]]]],"bias":[-3.4,-3.0],"names":["hole_strict","hole_lenient"]},
{"type":"relu"},{"type":"maxpool","size":7},{"type":"flatten"}],
"neurons":["strict hole top-left","strict hole top-right","strict hole bottom-left","strict hole bottom-right","lenient hole top-left","lenient hole top-right","lenient hole bottom-left","lenient hole bottom-right"]},
{"name":"ink_endpoints4x4","author":"Claude Opus 5.5","notes":"endpoint = relu(center3x3 - 1.5*ring5x5 - 0.5); ink = pixel; avgpool7 -> 4x4 zones",
"layers":[
{"type":"conv2d","padding":2,"weights":[[[[-1.5,-1.5,-1.5,-1.5,-1.5],[-1.5,1.0,1.0,1.0,-1.5],[-1.5,1.0,1.0,1.0,-1.5],[-1.5,1.0,1.0,1.0,-1.5],[-1.5,-1.5,-1.5,-1.5,-1.5]]],[[[0,0,0,0,0],[0,0,0,0,0],[0,0,1,0,0],[0,0,0,0,0],[0,0,0,0,0]]]],"bias":[-0.5,0],"names":["endpoint","ink"]},
{"type":"relu"},{"type":"avgpool","size":7},{"type":"flatten"}],
"neurons":["endpoint r0c0","endpoint r0c1","endpoint r0c2","endpoint r0c3","endpoint r1c0","endpoint r1c1","endpoint r1c2","endpoint r1c3","endpoint r2c0","endpoint r2c1","endpoint r2c2","endpoint r2c3","endpoint r3c0","endpoint r3c1","endpoint r3c2","endpoint r3c3","ink r0c0","ink r0c1","ink r0c2","ink r0c3","ink r1c0","ink r1c1","ink r1c2","ink r1c3","ink r2c0","ink r2c1","ink r2c2","ink r2c3","ink r3c0","ink r3c1","ink r3c2","ink r3c3"]}
]}
```