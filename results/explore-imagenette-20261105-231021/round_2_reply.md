Round 1 showed that the bank is weak overall. The class means barely differ for most neurons, and the four quadrant versions of each neuron are 70–98% correlated, so they mostly repeat each other. Some signals are clear:
- **Parachute:** sky blue and cyan, with the lowest edge energy.
- **Church:** blue in the top half.
- **Golf ball:** green.
- **Cassette player and chain saw:** white backgrounds.
- **French horn:** the most fine texture.

The bank still can't tell apart tench, English springer, garbage truck or gas pump.

The final layer is linear. So instead of single hand-tuned thresholds, I'm building histogram-like features: each neuron is the share of pixels above one threshold, and several thresholds per measure let the final layer pick out bands. Each colour score is the gap between one channel and the larger of the other two, worked out exactly by two relu layers.

- **R2F1: global colour histogram (40 neurons).** Brightness at 6 thresholds, saturation at 6, red, yellow, green and blue at 4 thresholds each, plus cyan, magenta and orange. It also counts light-grey, dark-grey or black, low-saturation, dark and black pixels.
- **R2F2: spatial layout (40 neurons).** Eight colour detectors (sky blue, green, low saturation, white, dark, warm yellow, red, saturated) averaged over each of four horizontal bands and over the centre. A dense layer combines the 4×4 averages into these bands. The aim is to find sky at the top and grass or road at the bottom, and to compare the object in the centre with its surroundings.
- **R2F3: texture histogram at half resolution (40 neurons).** I shrink to 32×32 and measure edge strength in four directions at several thresholds. It also measures:
  - fine texture at 5 thresholds;
  - how much of the image is flat;
  - very strong edges;
  - areas where horizontal or vertical lines dominate;
  - small spots;
  - straight up-down or sideways edges (man-made objects) versus diagonal ones;
  - red–green and blue–yellow colour edges.

```json
{"name":"R2F1_colour_histogram","author":"Claude Opus 5.5","notes":"Layer1 1x1: R,G,B, relu(R-G),relu(G-R),relu(G-B),relu(B-G),relu(R-B),relu(B-R),relu(R+B-2G). Layer2 1x1: exact hue minima e.g. red=min(R-G,R-B)=R-G-relu(B-G), saturation=(sum of abs diffs)/2, each at several thresholds, relu, global mean = fraction-weighted histogram.","neurons":["mean_bright","bright>0.15","bright>0.3","bright>0.45","bright>0.6","bright>0.75","bright>0.9","sat>0.05","sat>0.12","sat>0.2","sat>0.3","sat>0.45","sat>0.6","red>0.05","red>0.15","red>0.3","red>0.5","yellow>0.05","yellow>0.15","yellow>0.3","yellow>0.45","green>0.02","green>0.08","green>0.15","green>0.25","blue>0.02","blue>0.08","blue>0.15","blue>0.3","cyan>0.05","cyan>0.15","magenta>0.03","magenta>0.12","orange>0.05","orange>0.15","light_gray","dark_gray_black","low_sat","dark<0.2","black<0.08"],
"layers":[{"type":"conv2d","padding":0,"weights":"@hsv_l1","bias":[0,0,0,0,0,0,0,0,0,0]},{"type":"relu"},{"type":"conv2d","padding":0,"weights":"@hsv_l2","bias":[0,-0.15,-0.3,-0.45,-0.6,-0.75,-0.9,-0.05,-0.12,-0.2,-0.3,-0.45,-0.6,-0.05,-0.15,-0.3,-0.5,-0.05,-0.15,-0.3,-0.45,-0.02,-0.08,-0.15,-0.25,-0.02,-0.08,-0.15,-0.3,-0.05,-0.15,-0.03,-0.12,-0.05,-0.15,-0.55,0.3,0.06,0.2,0.08]},{"type":"relu"},{"type":"avgpool","size":64},{"type":"flatten"}]}
```

```json
{"name":"R2F2_colour_layout_bands","author":"Claude Opus 5.5","notes":"Same exact-hue layer1, then 8 detectors (sky_blue, green, low_sat, white, dark, warm_yellow, red, saturated), relu, avgpool16 -> 4x4, dense averages each detector over horizontal bands row0..row3 and the centre 2x2.","neurons":["sky_blue_row0","sky_blue_row1","sky_blue_row2","sky_blue_row3","sky_blue_center","green_row0","green_row1","green_row2","green_row3","green_center","low_sat_row0","low_sat_row1","low_sat_row2","low_sat_row3","low_sat_center","white_row0","white_row1","white_row2","white_row3","white_center","dark_row0","dark_row1","dark_row2","dark_row3","dark_center","warm_yellow_row0","warm_yellow_row1","warm_yellow_row2","warm_yellow_row3","warm_yellow_center","red_row0","red_row1","red_row2","red_row3","red_center","saturated_row0","saturated_row1","saturated_row2","saturated_row3","saturated_center"],
"layers":[{"type":"conv2d","padding":0,"weights":"@hsv_l1","bias":[0,0,0,0,0,0,0,0,0,0]},{"type":"relu"},{"type":"conv2d","padding":0,"weights":"@sp_l2","bias":[-0.05,-0.05,0.08,-0.75,0.25,-0.1,-0.1,-0.3],"names":["sky_blue","green","low_sat","white","dark","warm_yellow","red","saturated"]},{"type":"relu"},{"type":"avgpool","size":16},{"type":"flatten"},{"type":"dense","weights":"@sp_dense","bias":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}]}
```

```json
{"name":"R2F3_texture_histogram_scale2","author":"Claude Opus 5.5","notes":"avgpool2 -> 32x32; 3x3 grey sobel x/y, diag45, diag135, laplacian plus RG and BY opponent sobels; abs; 1x1 thresholded combinations (edge-strength histogram, flatness, horizontal/vertical dominance, axis-aligned vs diagonal, spots, colour edges); relu; global mean.","neurons":["|sx|>0.02","|sx|>0.06","|sx|>0.12","|sx|>0.25","|sy|>0.02","|sy|>0.06","|sy|>0.12","|sy|>0.25","|d45|>0.02","|d45|>0.06","|d45|>0.12","|d45|>0.25","|d135|>0.02","|d135|>0.06","|d135|>0.12","|d135|>0.25","|lap|>0.01","|lap|>0.03","|lap|>0.06","|lap|>0.12","|lap|>0.2","flat_grad<0.02","flat_grad<0.05","flat_grad<0.1","grad>0.25","grad>0.5","horiz_dominant_0.02","horiz_dominant_0.08","vert_dominant_0.02","vert_dominant_0.08","spot_0.01","spot_0.04","axis_aligned_0.02","axis_aligned_0.08","diagonal_0.02","diagonal_0.08","RGedge>0.03","RGedge>0.1","BYedge>0.03","BYedge>0.1"],
"layers":[{"type":"avgpool","size":2},{"type":"conv2d","padding":0,"weights":"@tx_c1","bias":[0,0,0,0,0,0,0,0,0]},{"type":"abs"},{"type":"conv2d","padding":0,"weights":"@tx_c2","bias":[-0.02,-0.06,-0.12,-0.25,-0.02,-0.06,-0.12,-0.25,-0.02,-0.06,-0.12,-0.25,-0.02,-0.06,-0.12,-0.25,-0.01,-0.03,-0.06,-0.12,-0.2,0.02,0.05,0.1,-0.25,-0.5,-0.02,-0.08,-0.02,-0.08,-0.01,-0.04,-0.02,-0.08,-0.02,-0.08,-0.03,-0.1,-0.03,-0.1]},{"type":"relu"},{"type":"avgpool","size":30},{"type":"flatten"}]}
```