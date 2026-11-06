I built three feature files for this first explore round. Together they measure where colours appear in the photo, how much texture or edge there is, and whether certain shapes are present anywhere. None of them is a classifier yet.

**R1F1: colour by quadrant (40 neurons).** A 1×1 conv turns each pixel into 10 colour scores, then I take the average of each score over each quarter of the image (top-left, top-right, bottom-left, bottom-right). What I expect each colour to pick up:
- **Sky blue:** sky in the top quarters for parachute and church; water for tench.
- **Green:** grass in the bottom quarters for golf ball, springer and tench.
- **Red:** gas pump, garbage truck and chain saw.
- **Yellow:** tench, yellow golf balls and chain saws.
- **White:** golf balls and cassette players shot on white backgrounds.
- **Dark:** chain saw and French horn photos with dark backgrounds.
- **Warm, brightness, magenta, cyan:** general scene cues that should help separate the rest.

**R1F2: fine-scale edges by quadrant (40 neurons).** These are 3×3 edge filters on the full 64×64 image, with absolute values averaged over each quarter. They cover:
- Edges in grey: horizontal, vertical, both diagonals, a texture filter and a thin horizontal-line detector.
- Edges where colour changes (red against green, blue against yellow), in both directions.

My guesses: open sky (parachute) should give very low edge scores. Grass and dimpled golf balls should give high texture scores. Trucks, cassette players and churches should have strong horizontal and vertical lines.

**R1F3: shapes anywhere in the image (15 neurons).** I shrink the photo to 16×16, run 5×5 filters, and keep each filter's strongest response anywhere in the frame. The filters look for:
- a bright blob, a dark blob, and a white blob on a darker surround (golf ball);
- a red blob and a yellow blob;
- the most blue patch and the most green patch;
- edges in four directions, including bright-above-dark (sky over ground);
- a bright vertical bar (church spire, gas pump) and a bright horizontal bar;
- a coloured blob surrounded by sky (parachute canopy);
- a cyan blob.

```json
{"name":"R1F1_colour_quadrants","author":"Claude Opus 5.5","notes":"1x1 colour-opponent detectors, relu, mean over each image quadrant. Order channel-major: each colour then TL,TR,BL,BR.","neurons":["sky_blue_TL","sky_blue_TR","sky_blue_BL","sky_blue_BR","green_TL","green_TR","green_BL","green_BR","red_TL","red_TR","red_BL","red_BR","yellow_TL","yellow_TR","yellow_BL","yellow_BR","white_TL","white_TR","white_BL","white_BR","dark_TL","dark_TR","dark_BL","dark_BR","warm_TL","warm_TR","warm_BL","warm_BR","bright_TL","bright_TR","bright_BL","bright_BR","magenta_TL","magenta_TR","magenta_BL","magenta_BR","cyan_TL","cyan_TR","cyan_BL","cyan_BR"],
"layers":[{"type":"conv2d","padding":0,"weights":"@f1w","bias":[-0.1,-0.03,-0.15,-0.15,-2.4,0.75,-0.1,0,-0.05,-0.1],"names":["sky_blue","green","red","yellow","white","dark","warm","bright","magenta","cyan"]},{"type":"relu"},{"type":"avgpool","size":32},{"type":"flatten"}]}
```

```json
{"name":"R1F2_edges_quadrants","author":"Claude Opus 5.5","notes":"3x3 edge/texture filters at full res (no padding, 62x62), abs, mean over 31x31 quadrants. Grey filters plus R-G and B-Y opponent edges.","neurons":["sobelx_TL","sobelx_TR","sobelx_BL","sobelx_BR","sobely_TL","sobely_TR","sobely_BL","sobely_BR","diag45_TL","diag45_TR","diag45_BL","diag45_BR","diag135_TL","diag135_TR","diag135_BL","diag135_BR","laplacian_TL","laplacian_TR","laplacian_BL","laplacian_BR","RGedge_x_TL","RGedge_x_TR","RGedge_x_BL","RGedge_x_BR","RGedge_y_TL","RGedge_y_TR","RGedge_y_BL","RGedge_y_BR","BYedge_x_TL","BYedge_x_TR","BYedge_x_BL","BYedge_x_BR","BYedge_y_TL","BYedge_y_TR","BYedge_y_BL","BYedge_y_BR","hline_TL","hline_TR","hline_BL","hline_BR"],
"layers":[{"type":"conv2d","padding":0,"weights":"@f2w","bias":[0,0,0,0,0,0,0,0,0,0],"names":["sobelx","sobely","diag45","diag135","laplacian","RGedge_x","RGedge_y","BYedge_x","BYedge_y","hline"]},{"type":"abs"},{"type":"avgpool","size":31},{"type":"flatten"}]}
```

```json
{"name":"R1F3_coarse_blobs_max","author":"Claude Opus 5.5","notes":"avgpool4 -> 16x16, 5x5 centre-surround / bar / edge / colour-blob filters (pad 0 -> 12x12), relu, global max.","neurons":["bright_blob_center_vs_ring","dark_blob_center_vs_ring","white_blob_golfball","red_blob","yellow_blob","max_blue_patch","max_green_patch","vert_edge_left_bright","vert_edge_right_bright","horiz_edge_top_bright_sky_over_ground","horiz_edge_bottom_bright","vertical_bright_bar_spire_pump","horizontal_bright_bar","warm_blob_in_blue_surround_canopy","cyan_blob"],
"layers":[{"type":"avgpool","size":4},{"type":"conv2d","padding":0,"weights":"@f3w","bias":[-0.15,-0.15,-0.8,-0.3,-0.3,-0.2,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.3,-0.25],"names":["bright_blob","dark_blob","white_blob","red_blob","yellow_blob","blue_patch","green_patch","vedge_pos","vedge_neg","hedge_topbright","hedge_botbright","vbar","hbar","canopy","cyan_blob"]},{"type":"relu"},{"type":"maxpool","size":12},{"type":"flatten"}]}
```