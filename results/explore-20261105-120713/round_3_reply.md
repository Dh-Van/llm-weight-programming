This round adds three feature files to fix weak spots in the measurements.

**What last round showed**
- The lenient hole neurons (R2F2.4–7) also fire on 2, 3 and 7 (means 0.2–0.4). The strict ones are cleaner but still give 0.07–0.17 on those digits. The four-arm test is fooled by concave shapes that are open diagonally.
- The 3×3 stroke zones are too coarse. Many stroke and endpoint neurons barely separate the digits. The 4×4 ink zones do separate them, so finer position information helps.
- The pairs the bank still can't reliably tell apart are 4/9 (closed vs open top), 3/5 (which side the top-left stroke is on), 7/9, and 2/3/7, which partly fire the hole neurons.

**New files**
1. **R3F1, 8-direction hole detector (8 neurons).** On the 14×14 downsample, it needs capped ink along all 8 arms: four straight arms 4 px long and four diagonal arms 3 px long. The centre pixel is penalised by 3. Thresholds are 7.4 (strict) and 6.6 (lenient), max-pooled into four quadrants. Prediction: the top quadrants fire for 0, 8, 9 and closed 4s, the bottom ones for 0, 6 and 8, and 2, 3, 5 and 7 drop close to 0.
2. **R3F2, ink density on a 6×6 grid of overlapping 8×8 blocks with a 4-pixel step (36 neurons).** This is a finer template map for the combine step's linear layer.
3. **R3F3, horizontal and vertical strokes in 4×4 zones of 7×7 pixels (32 neurons).** Predictions:
   - top-row horizontal neurons pick out 7, 5 and 2;
   - bottom-row horizontal neurons pick out 2;
   - left-column vertical neurons pick out 0 and 6;
   - middle-column vertical neurons pick out 1;
   - top-left vertical (5) versus top-right vertical (3) should help split 3 from 5.

```json
{"name":"holes8dir","author":"Claude Opus 5.5","notes":"14x14 image; 8 arms (4 axial len4, 4 diagonal len3), each clipped to 1 via relu(a)-relu(a-1); hole = relu(sum - 3*center - thr), thr 7.4 strict / 6.6 lenient; maxpool7 quadrants",
"layers":[
{"type":"avgpool","size":2},
{"type":"conv2d","padding":4,"weights":[[[[0,0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0]]],[[[0,0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0]]],[[[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0,0]]],[[[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0,0]]],[[[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[1,1,1,1,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0]]],[[[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[1,1,1,1,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0]]],[[[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,1,1,1,1],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0]]],[[[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,1,1,1,1],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0]]],[[[0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0]]],[[[0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0]]],[[[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,1,0,0],[0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0]]],[[[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,1,0,0],[0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0]]],[[[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0],[0,0,1,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0]]],[[[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0],[0,0,1,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0]]],[[[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,0]]],[[[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,0]]],[[[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0]]]],
"bias":[0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0]},
{"type":"relu"},
{"type":"conv2d","padding":0,"weights":[[[[1]],[[-1]],[[1]],[[-1]],[[1]],[[-1]],[[1]],[[-1]],[[1]],[[-1]],[[1]],[[-1]],[[1]],[[-1]],[[1]],[[-1]],[[-3]]],[[[1]],[[-1]],[[1]],[[-1]],[[1]],[[-1]],[[1]],[[-1]],[[1]],[[-1]],[[1]],[[-1]],[[1]],[[-1]],[[1]],[[-1]],[[-3]]]],"bias":[-7.4,-6.6],"names":["hole8_strict","hole8_lenient"]},
{"type":"relu"},{"type":"maxpool","size":7},{"type":"flatten"}],
"neurons":["8dir strict hole top-left","8dir strict hole top-right","8dir strict hole bottom-left","8dir strict hole bottom-right","8dir lenient hole top-left","8dir lenient hole top-right","8dir lenient hole bottom-left","8dir lenient hole bottom-right"]}
```

```json
{"name":"ink6x6","author":"Claude Opus 5.5","notes":"avgpool4 -> 7x7, then 2x2 averaging conv -> 6x6 grid of overlapping 8x8 pixel blocks (stride 4)",
"layers":[
{"type":"avgpool","size":4},
{"type":"conv2d","padding":0,"weights":[[[[0.25,0.25],[0.25,0.25]]]],"bias":[0]},
{"type":"flatten"}],
"neurons":["ink r0c0","ink r0c1","ink r0c2","ink r0c3","ink r0c4","ink r0c5","ink r1c0","ink r1c1","ink r1c2","ink r1c3","ink r1c4","ink r1c5","ink r2c0","ink r2c1","ink r2c2","ink r2c3","ink r2c4","ink r2c5","ink r3c0","ink r3c1","ink r3c2","ink r3c3","ink r3c4","ink r3c5","ink r4c0","ink r4c1","ink r4c2","ink r4c3","ink r4c4","ink r4c5","ink r5c0","ink r5c1","ink r5c2","ink r5c3","ink r5c4","ink r5c5"]}
```

```json
{"name":"hv_strokes4x4","author":"Claude Opus 5.5","notes":"zero-sum 5x5 horizontal and vertical stroke detectors, bias -0.5, relu, avgpool7 -> 4x4 zones each",
"layers":[
{"type":"conv2d","padding":2,"weights":[[[[-0.3,-0.3,-0.3,-0.3,-0.3],[0.2,0.2,0.2,0.2,0.2],[0.2,0.2,0.2,0.2,0.2],[0.2,0.2,0.2,0.2,0.2],[-0.3,-0.3,-0.3,-0.3,-0.3]]],[[[-0.3,0.2,0.2,0.2,-0.3],[-0.3,0.2,0.2,0.2,-0.3],[-0.3,0.2,0.2,0.2,-0.3],[-0.3,0.2,0.2,0.2,-0.3],[-0.3,0.2,0.2,0.2,-0.3]]]],"bias":[-0.5,-0.5],"names":["horiz","vert"]},
{"type":"relu"},{"type":"avgpool","size":7},{"type":"flatten"}],
"neurons":["horiz r0c0","horiz r0c1","horiz r0c2","horiz r0c3","horiz r1c0","horiz r1c1","horiz r1c2","horiz r1c3","horiz r2c0","horiz r2c1","horiz r2c2","horiz r2c3","horiz r3c0","horiz r3c1","horiz r3c2","horiz r3c3","vert r0c0","vert r0c1","vert r0c2","vert r0c3","vert r1c0","vert r1c1","vert r1c2","vert r1c3","vert r2c0","vert r2c1","vert r2c2","vert r2c3","vert r3c0","vert r3c1","vert r3c2","vert r3c3"]}
```