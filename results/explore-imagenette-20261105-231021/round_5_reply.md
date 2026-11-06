This is the last explore round. R4F1 works, but its outputs are too small: most class means show as 0.00–0.06, so the tables don't give me enough precision to weight them. R4F2 (full resolution) and R4F3 (16×16) texture came out well scaled. They add some signal on top of the half-resolution texture:
- **Coarse texture:** strong edges for springer and cassette player, the most blue–yellow edges in the bottom half for tench, horizontal lines for cassette player and garbage truck, vertical lines for gas pump and springer.
- **Fine texture:** French horn texture and gas-pump vertical lines.

What's still weakest is telling tench, springer and garbage truck apart by object colour.

- **R5F1: R4F1 rescaled.** It reuses all 10 layers of R4F1 unchanged and adds a diagonal dense layer that divides each output by its measured spread. The neurons are the same, but their means will now be reported to usable precision.
- **R5F2: 20 finer colour categories, whole image and centre (40 neurons).** Each category requires two conditions at once (a two-condition minimum built from relus), and its output is clipped to [0,1], so each neuron is close to the fraction of pixels in that category. Scaled ×5 (whole image) and ×4 (centre). The categories and what I expect them to pick up:
  - dark brown, tan and skin (springer coat, people, tench);
  - olive (tench in water) and gold/brass (French horn);
  - dark green and light green (foliage, sunny grass);
  - pale blue sky and deep blue;
  - teal (garbage trucks, water);
  - pink;
  - bright red and dark red (gas pump, chain saw, fire-engine red);
  - bright orange and bright yellow (chain saw);
  - pure white, black, mid/light grey, dark grey, and silver metal (cassette player, French horn).
- **R5F3: the same 20 categories, strongest patch anywhere (20 neurons, ×2).** For each category I take the largest share of matching pixels in any 8×8 block, which detects whether an object of that colour appears somewhere in the frame. Examples: a red pump body, an orange saw body, a white ball, a brass horn, a brown dog patch.

```json
{"name":"R5F1_conjunctions_rescaled","author":"Claude Opus 5.5","notes":"Runs R4F1 unchanged, then a diagonal dense layer dividing each neuron by its measured std (0.05,0.21,0.23,0.23,0.04,0.04,0.18,0.18,0.22,0.20,0.24,0.13,0.16,0.25,0.39,0.18,0.41,0.40,0.34,0.18) so outputs have unit spread.","base":"R4F1","base_layers":10,"neurons":["sky_above_nonsky","sky_above_green","white_on_green_golf","colour_in_sky_canopy","bright_spot","dark_spot","vedge_left_bright","vedge_right_bright","hedge_top_bright","hedge_bottom_bright","red_next_to_white","yellow_in_dark","dark_on_white_bg","orange_next_white_springer","green_above_gray_street","orange_on_green_dog","gray_under_sky","colour_amid_gray","yellow_in_green","dark_silhouette_in_sky"],
"layers":[{"type":"dense","weights":"@r4f1_diag","bias":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}]}
```

```json
{"name":"R5F2_fine_colour_bins","author":"Claude Opus 5.5","notes":"Exact-hue layer1; layer2 gives relu(a) and relu(a-b) for two margin conditions a,b per category; layer3 makes clipped ramp of min(a,b)=relu(a)-relu(a-b) above threshold t with slope 20 (two relus, difference in layer4 gives value in [0,1]); avgpool16 -> 4x4; dense: global fraction x5 and centre 2x2 fraction x4.","neurons":["brown_dark_global","brown_dark_center","tan_light_brown_global","tan_light_brown_center","skin_global","skin_center","olive_global","olive_center","dark_green_global","dark_green_center","light_green_global","light_green_center","pale_blue_sky_global","pale_blue_sky_center","deep_blue_global","deep_blue_center","teal_global","teal_center","pink_global","pink_center","bright_red_global","bright_red_center","dark_red_global","dark_red_center","bright_orange_global","bright_orange_center","bright_yellow_global","bright_yellow_center","pure_white_global","pure_white_center","black_global","black_center","gray_mid_light_global","gray_mid_light_center","gray_dark_global","gray_dark_center","silver_metal_global","silver_metal_center","gold_brass_global","gold_brass_center"],
"layers":[{"type":"conv2d","padding":0,"weights":"@hsv_l1","bias":[0,0,0,0,0,0,0,0,0,0]},{"type":"relu"},{"type":"conv2d","padding":0,"weights":"@fc_l2","bias":[0,-0.45,0,0.45,0,-0.35,0,-0.1,0,-0.4,0,0.4,0,0.55,0,-0.5,0,-0.15,0,0.4,0,0.4,0,-0.5,0,0.6,0,0.5,-0.85,-0.93,0.1,-0.9,0.06,0.36,0.06,-0.24,0.1,0.7,0,0.25]},{"type":"relu"},{"type":"conv2d","padding":0,"weights":"@fc_l3","bias":[-0.6,-1.6,-0.6,-1.6,-0.6,-1.6,-0.4,-1.4,-0.4,-1.4,-0.4,-1.4,-0.4,-1.4,-1.0,-2.0,-1.0,-2.0,-1.0,-2.0,-4.0,-5.0,-2.0,-3.0,-2.0,-3.0,-2.0,-3.0,-0.1,-1.1,0,-1.0,-0.1,-1.1,-0.1,-1.1,-0.2,-1.2,-1.0,-2.0]},{"type":"relu"},{"type":"conv2d","padding":0,"weights":"@fc_l4","bias":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]},{"type":"avgpool","size":16},{"type":"flatten"},{"type":"dense","weights":"@fc_dense","bias":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}]}
```

```json
{"name":"R5F3_fine_colour_patch_max","author":"Claude Opus 5.5","notes":"Same clipped colour-category maps as R5F2, avgpool8 -> 8x8 grid of patch fractions, global maxpool -> strongest 8x8 patch of each colour anywhere, x2.","neurons":["brown_dark_max","tan_light_brown_max","skin_max","olive_max","dark_green_max","light_green_max","pale_blue_sky_max","deep_blue_max","teal_max","pink_max","bright_red_max","dark_red_max","bright_orange_max","bright_yellow_max","pure_white_max","black_max","gray_mid_light_max","gray_dark_max","silver_metal_max","gold_brass_max"],
"layers":[{"type":"conv2d","padding":0,"weights":"@hsv_l1","bias":[0,0,0,0,0,0,0,0,0,0]},{"type":"relu"},{"type":"conv2d","padding":0,"weights":"@fc_l2","bias":[0,-0.45,0,0.45,0,-0.35,0,-0.1,0,-0.4,0,0.4,0,0.55,0,-0.5,0,-0.15,0,0.4,0,0.4,0,-0.5,0,0.6,0,0.5,-0.85,-0.93,0.1,-0.9,0.06,0.36,0.06,-0.24,0.1,0.7,0,0.25]},{"type":"relu"},{"type":"conv2d","padding":0,"weights":"@fc_l3","bias":[-0.6,-1.6,-0.6,-1.6,-0.6,-1.6,-0.4,-1.4,-0.4,-1.4,-0.4,-1.4,-0.4,-1.4,-1.0,-2.0,-1.0,-2.0,-1.0,-2.0,-4.0,-5.0,-2.0,-3.0,-2.0,-3.0,-2.0,-3.0,-0.1,-1.1,0,-1.0,-0.1,-1.1,-0.1,-1.1,-0.2,-1.2,-1.0,-2.0]},{"type":"relu"},{"type":"conv2d","padding":0,"weights":"@fc_l4","bias":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]},{"type":"avgpool","size":8},{"type":"maxpool","size":8},{"type":"flatten"},{"type":"scale","value":2}]}
```