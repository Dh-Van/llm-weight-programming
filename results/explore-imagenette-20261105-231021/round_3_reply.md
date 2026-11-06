Most round-2 neurons behave as intended, but their scale is a problem. Many neurons have class means of 0.00–0.01 in the 2-decimal tables, so I can't build a good final layer from them. Most thresholds of the same measure are also 0.9+ correlated with each other.

What works so far:
- **Parachute:** blue and saturated colour everywhere, plus flat areas.
- **Church:** blue sky only in the top band.
- **Cassette player:** horizontal-dominant edges (42 of the top 50).
- **Gas pump:** vertical-dominant edges and red.
- **French horn:** fine texture and spots.
- **Golf ball:** green and yellow.
- **Cassette player and chain saw:** white areas.

Tench, English springer and garbage truck still have no clear feature.

This round:
- **R3F1: colour features, rescaled (40 neurons).** These are the 22 least redundant whole-image colour neurons plus 18 band neurons (top/bottom rows and centre). Each output is divided by its measured spread so its spread is about 1, which makes the class means precise enough to use in the final layer.
- **R3F2: texture by band, rescaled (40 neurons).** I take 14 texture measures at half resolution and pool them over a 3×3 grid. Each measure is reported for the whole image, the top band and the bottom band, and four of them also for the centre cell, scaled the same way. Examples: a busy top with a flat bottom, vertical lines in the centre, horizontal lines at the bottom.
- **R3F3: neighbouring-colour patterns (20 neurons, ×10).** I make 10 colour maps at 16×16, then use 3×3 filters to find colours next to each other. They target classes the bank can't separate yet:
  - sky above green, and sky above a non-sky area;
  - green trees above a grey road (garbage truck);
  - grey structure under sky (church or truck);
  - red next to white (gas pump signs);
  - a yellow object in dark or in green (tench, brass);
  - a dark object on a white background (product shots);
  - orange next to white, and orange on green (springer);
  - a white blob on green (golf ball);
  - a coloured shape in sky, and a dark shape in sky (parachute);
  - a colourful object surrounded by grey;
  - bright and dark spots, and large edges in four directions.

```json
{"name":"R3F1_colour_scaled","author":"Claude Opus 5.5","notes":"Exact-hue layer1; 30 detector maps (22 global thresholds + 8 band detectors); relu; avgpool16 -> 4x4; dense averages over whole image / rows / centre, each divided by its measured round-2 std so outputs have ~unit spread.","neurons":["mean_bright","bright>0.6","bright>0.9","sat>0.05","sat>0.3","sat>0.6","red>0.05","red>0.3","yellow>0.05","yellow>0.3","green>0.02","green>0.15","blue>0.02","blue>0.15","cyan>0.05","magenta>0.03","orange>0.05","orange>0.15","light_gray","dark_gray_black","low_sat","black","sky_blue_row0","sky_blue_row3","green_row0","green_row3","white_row0","white_row3","white_center","dark_row0","dark_row3","warm_yellow_row0","warm_yellow_row2","red_row0","red_row3","red_center","low_sat_row0","low_sat_row3","saturated_row0","saturated_row3"],
"layers":[{"type":"conv2d","padding":0,"weights":"@hsv_l1","bias":[0,0,0,0,0,0,0,0,0,0]},{"type":"relu"},{"type":"conv2d","padding":0,"weights":"@c3_l2","bias":[0,-0.6,-0.9,-0.05,-0.3,-0.6,-0.05,-0.3,-0.05,-0.3,-0.02,-0.15,-0.02,-0.15,-0.05,-0.03,-0.05,-0.15,-0.55,0.3,0.06,0.08,-0.05,-0.05,0.08,-0.75,0.25,-0.1,-0.1,-0.3]},{"type":"relu"},{"type":"avgpool","size":16},{"type":"flatten"},{"type":"dense","weights":"@c3_dense","bias":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}]}
```

```json
{"name":"R3F2_texture_bands_scaled","author":"Claude Opus 5.5","notes":"avgpool2; 3x3 grey sobels/diagonals/laplacian + RG/BY opponent sobels; abs; 14 thresholded texture detectors; relu; avgpool10 -> 3x3 grid; dense: global, top band, bottom band (and centre cell for some), scaled by 1/estimated std.","neurons":["grad>0.1_global","grad>0.1_top","grad>0.1_bottom","flat<0.05_global","flat<0.05_top","flat<0.05_bottom","horiz_dom_global","horiz_dom_top","horiz_dom_bottom","vert_dom_global","vert_dom_top","vert_dom_bottom","lap>0.03_global","lap>0.03_top","lap>0.03_bottom","spot_global","spot_top","spot_bottom","diagonal_global","diagonal_top","diagonal_bottom","axis_aligned_global","axis_aligned_top","axis_aligned_bottom","RGedge_global","RGedge_top","RGedge_bottom","BYedge_global","BYedge_top","BYedge_bottom","grad>0.5_global","grad>0.5_top","grad>0.5_bottom","grad>0.1_center","horiz_dom_center","vert_dom_center","lap>0.03_center","grad>0.25_global","lap>0.2_global","flat<0.02_global"],
"layers":[{"type":"avgpool","size":2},{"type":"conv2d","padding":0,"weights":"@tx_c1","bias":[0,0,0,0,0,0,0,0,0]},{"type":"abs"},{"type":"conv2d","padding":0,"weights":"@t3_c2","bias":[-0.1,0.05,-0.02,-0.02,-0.03,-0.01,-0.02,-0.02,-0.03,-0.03,-0.5,-0.25,-0.2,0.02]},{"type":"relu"},{"type":"avgpool","size":10},{"type":"flatten"},{"type":"dense","weights":"@t3_dense","bias":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}]}
```

```json
{"name":"R3F3_colour_conjunctions","author":"Claude Opus 5.5","notes":"10 colour maps scaled to ~[0,1] (sky,green,low_sat,white,dark,warm_yellow,red,saturated,brightness,orange) -> avgpool4 (16x16) -> 3x3 conjunction filters (A in centre/top + B in ring/bottom minus threshold) -> relu -> global mean -> x10.","neurons":["sky_above_nonsky","sky_above_green","white_on_green_golf","colour_in_sky_canopy","bright_spot","dark_spot","vedge_left_bright","vedge_right_bright","hedge_top_bright","hedge_bottom_bright","red_next_to_white","yellow_in_dark","dark_on_white_bg","orange_next_white_springer","green_above_gray_street","orange_on_green_dog","gray_under_sky","colour_amid_gray","yellow_in_green","dark_silhouette_in_sky"],
"layers":[{"type":"conv2d","padding":0,"weights":"@hsv_l1","bias":[0,0,0,0,0,0,0,0,0,0]},{"type":"relu"},{"type":"conv2d","padding":0,"weights":"@cj_l2","bias":[-0.1,-0.09,0.8,-3.0,0.75,-0.16,-0.15,-0.6,0,-0.2]},{"type":"relu"},{"type":"avgpool","size":4},{"type":"conv2d","padding":0,"weights":"@cj_c3","bias":[-0.2,-0.6,-0.7,-0.5,-0.15,-0.15,-0.1,-0.1,-0.1,-0.1,-0.6,-0.6,-0.7,-0.6,-0.6,-0.5,-0.6,-0.6,-0.4,-0.6]},{"type":"relu"},{"type":"avgpool","size":14},{"type":"flatten"},{"type":"scale","factor":10}]}
```