[Round 7 of 7: COMBINE. This is the LAST round: submit your final classifier.]

Class numbers: 0 = tench, 1 = English springer, 2 = cassette player, 3 = chain saw, 4 = church, 5 = French horn, 6 = garbage truck, 7 = gas pump, 8 = golf ball, 9 = parachute.
Results of your classifier on the validation split:

accuracy: 69.04%

per-class accuracy:
  0:  81.2%
  1:  71.0%
  2:  68.9%
  3:  52.3%
  4:  66.7%
  5:  69.2%
  6:  73.8%
  7:  66.5%
  8:  66.0%
  9:  74.0%

confusion matrix (row = true, col = predicted):
           0     1     2     3     4     5     6     7     8     9
   0     397    22     3    11     9    11     9     2    22     3
   1      20   341     9    40    12    13     5    10    24     6
   2       7    21   366    34    17    16    22    32     8     8
   3      37    33    21   245    17    50    25    18    12    10
   4      16     6     6    20   344    32    34    31    18     9
   5      18    32    13    39    19   349     4    20     7     3
   6       9    11    22    26    24     9   371    21     8     2
   7       9    12    17    31    33    10    39   321     5     6
   8      30    31    11    25    16    15    13    10   340    24
   9      14     7     4    26    22     3    29     6    22   378

Your most frequent confusions, explained through your final layer (averaged over all validation images with that confusion; the terms plus the bias gap sum to the score gap):
```
3 (chain saw) predicted as 5 (French horn): 50 images. Average score[3] - score[5] = -2.084 (sum of neuron terms -8.024, bias gap +5.940)
  neurons pushing toward 5 (average (w[3]-w[5]) x value):
    R4F2.17 (fine_spot_bottom): value 1.140, w[3] -0.209, w[5] +1.950 -> -2.462
    R4F2.15 (fine_spot_global): value 1.460, w[3] -0.195, w[5] +1.080 -> -1.861
    R3F2.15 (spot_global): value 2.054, w[3] -0.229, w[5] +0.499 -> -1.496
    R4F2.20 (fine_diagonal_bottom): value 1.639, w[3] -0.150, w[5] +0.486 -> -1.043
    R3F2.18 (diagonal_global): value 2.333, w[3] -0.264, w[5] +0.142 -> -0.947
    R3F2.36 (lap>0.03_center): value 2.704, w[3] +0.083, w[5] +0.402 -> -0.862
  neurons pushing toward 3: R4F2.36 (fine_lap>0.03_left) +1.266; R4F2.13 (fine_lap>0.03_top) +0.858; R3F2.13 (lap>0.03_top) +0.831

1 (English springer) predicted as 3 (chain saw): 40 images. Average score[1] - score[3] = -1.472 (sum of neuron terms -3.022, bias gap +1.550)
  neurons pushing toward 3 (average (w[1]-w[3]) x value):
    R4F2.12 (fine_lap>0.03_global): value 1.713, w[1] -0.421, w[3] +0.264 -> -1.173
    R4F2.36 (fine_lap>0.03_left): value 1.288, w[1] -0.469, w[3] +0.380 -> -1.094
    R5F3.10 (bright_red_max): value 0.726, w[1] -0.488, w[3] +0.957 -> -1.049
    R4F2.13 (fine_lap>0.03_top): value 1.477, w[1] -0.401, w[3] +0.240 -> -0.947
    R4F2.37 (fine_lap>0.03_right): value 1.347, w[1] -0.479, w[3] +0.220 -> -0.942
    R4F2.14 (fine_lap>0.03_bottom): value 1.159, w[1] -0.464, w[3] +0.301 -> -0.887
  neurons pushing toward 1: R3F2.33 (grad>0.1_center) +1.369; R1F3.11 (vertical_bright_bar_spire_pump) +0.962; R3F2.18 (diagonal_global) +0.687

5 (French horn) predicted as 3 (chain saw): 39 images. Average score[5] - score[3] = -1.779 (sum of neuron terms +4.161, bias gap -5.940)
  neurons pushing toward 3 (average (w[5]-w[3]) x value):
    R5F3.10 (bright_red_max): value 0.899, w[5] -0.225, w[3] +0.957 -> -1.063
    R4F2.36 (fine_lap>0.03_left): value 1.384, w[5] -0.239, w[3] +0.380 -> -0.857
    R3F2.33 (grad>0.1_center): value 2.001, w[5] -0.205, w[3] +0.118 -> -0.646
    R3F2.13 (lap>0.03_top): value 1.919, w[5] -0.114, w[3] +0.194 -> -0.591
    R4F2.13 (fine_lap>0.03_top): value 1.426, w[5] -0.169, w[3] +0.240 -> -0.583
    R5F3.12 (bright_orange_max): value 0.617, w[5] -0.289, w[3] +0.544 -> -0.514
  neurons pushing toward 5: R4F2.17 (fine_spot_bottom) +1.349; R4F2.15 (fine_spot_global) +1.142; R3F2.15 (spot_global) +1.039

7 (gas pump) predicted as 6 (garbage truck): 39 images. Average score[7] - score[6] = -2.129 (sum of neuron terms -2.199, bias gap +0.070)
  neurons pushing toward 6 (average (w[7]-w[6]) x value):
    R4F2.8 (fine_horiz_dom_bottom): value 1.390, w[7] -0.238, w[6] +0.774 -> -1.407
    R5F3.17 (gray_dark_max): value 1.188, w[7] +0.007, w[6] +0.716 -> -0.843
    R3F1.37 (low_sat_row3): value 1.746, w[7] +0.022, w[6] +0.443 -> -0.734
    R4F2.0 (fine_grad>0.1_global): value 2.241, w[7] -0.107, w[6] +0.197 -> -0.681
    R4F2.2 (fine_grad>0.1_bottom): value 1.612, w[7] -0.150, w[6] +0.236 -> -0.622
    R4F3.8 (coarse_horiz_dom_bottom): value 1.840, w[7] -0.131, w[6] +0.172 -> -0.557
  neurons pushing toward 7: R4F2.7 (fine_horiz_dom_top) +1.049; R4F2.11 (fine_vert_dom_bottom) +0.993; R4F2.22 (fine_axis_aligned_top) +0.855

3 (chain saw) predicted as 0 (tench): 37 images. Average score[3] - score[0] = -1.624 (sum of neuron terms -1.424, bias gap -0.200)
  neurons pushing toward 0 (average (w[3]-w[0]) x value):
    R5F3.4 (dark_green_max): value 1.149, w[3] -0.184, w[0] +1.150 -> -1.532
    R4F3.29 (coarse_BYedge_bottom): value 2.073, w[3] -0.035, w[0] +0.526 -> -1.163
    R5F3.5 (light_green_max): value 0.943, w[3] +0.169, w[0] +1.140 -> -0.916
    R5F2.3 (tan_light_brown_center): value 0.575, w[3] -0.060, w[0] +1.300 -> -0.782
    R3F1.32 (warm_yellow_row2): value 0.961, w[3] -0.134, w[0] +0.560 -> -0.667
    R5F2.27 (bright_yellow_center): value 0.415, w[3] -0.505, w[0] +1.080 -> -0.658
  neurons pushing toward 3: R3F2.36 (lap>0.03_center) +0.772; R5F2.2 (tan_light_brown_global) +0.562; R4F2.36 (fine_lap>0.03_left) +0.533
```

10 individual mistakes:

true 7 (gas pump), predicted 6 (garbage truck)
scores [0: -3.50, 1: -6.57, 2: -5.40, 3: -2.66, 4: -3.71, 5: -4.88, 6: -2.55, 7: -3.73, 8: -7.81, 9: -10.64]
pixels (the photo shrunk to 16x16 (each cell is the average of a 4x4 block): 16 rows of 16 cells separated by spaces, each cell 3 hex digits = red, green, blue, 0 (dark) to f (bright); the network itself sees the full-resolution photo):
```
99a 9aa a9a ddc ffe bcc 767 988 877 888 777 888 777 766 665 888
aaa a9a eff bbb ddc ccc aab 778 766 877 777 877 766 665 555 666
eee bbc eee a9a ccb dcc a99 999 777 776 877 877 554 655 655 777
fff bcc ddd a99 aa9 cbb aaa 988 766 665 766 777 555 767 666 655
a9a a9a ddd ccc 766 977 877 a88 ba9 877 555 778 777 555 555 555
866 967 baa baa a88 988 877 987 a98 a99 865 865 965 755 865 887
a79 a68 bab baa 987 987 987 987 a98 a98 744 754 854 767 866 aa8
545 666 999 998 886 987 877 876 997 997 989 888 877 877 877 987
433 443 655 877 a97 987 876 976 987 986 aa9 abc aac 9aa 876 a97
333 433 655 444 543 533 544 654 643 543 877 9ac 89c 9ab 765 654
333 333 654 755 644 655 544 755 654 543 655 9bd 9ac 99a 865 543
555 655 765 866 976 765 988 888 777 555 554 334 455 888 988 776
aaa 876 876 765 765 765 667 567 457 456 345 234 222 766 988 333
99a 778 555 444 444 433 433 444 445 445 556 777 999 667 777 555
567 667 567 556 455 444 444 655 755 876 877 888 778 456 445 766
556 567 567 567 567 888 999 777 777 544 443 443 555 444 333 333
```
biggest terms toward 6: R4F2.8 (fine_horiz_dom_bottom) value 1.27 -> -1.284; R4F2.0 (fine_grad>0.1_global) value 2.46 -> -0.748; R4F2.2 (fine_grad>0.1_bottom) value 1.73 -> -0.670; R4F3.8 (coarse_horiz_dom_bottom) value 2.13 -> -0.644; R4F3.2 (coarse_grad>0.1_bottom) value 2.70 -> -0.596

true 2 (cassette player), predicted 3 (chain saw)
scores [0: -3.35, 1: -4.22, 2: -0.87, 3: -0.47, 4: -7.18, 5: -3.60, 6: -7.83, 7: -8.09, 8: -4.39, 9: -11.56]
pixels (the photo shrunk to 16x16 (each cell is the average of a 4x4 block): 16 rows of 16 cells separated by spaces, each cell 3 hex digits = red, green, blue, 0 (dark) to f (bright); the network itself sees the full-resolution photo):
```
211 111 111 111 211 211 211 211 222 222 222 211 211 111 111 111
111 111 211 222 212 222 222 222 111 322 222 111 111 211 211 221
111 211 222 322 655 766 766 655 222 443 443 322 221 211 321 321
322 433 877 dcb edc edc dcb dcb a99 765 533 332 322 322 432 432
544 766 987 ba9 edb edc dcb dcb cbb b87 965 965 753 543 432 322
544 655 544 444 444 555 776 988 baa b99 ca8 da8 c98 b97 d97 c86
555 665 433 221 333 544 655 554 544 544 877 b98 c97 c98 ca9 643
555 444 111 221 221 432 333 443 443 444 555 987 ca8 ca8 876 655
655 544 322 333 222 222 544 655 443 443 655 766 b98 655 433 433
555 544 666 666 655 545 544 544 655 887 544 544 321 211 311 544
655 655 655 655 555 655 666 554 655 655 544 433 322 322 544 877
555 767 766 665 655 555 544 544 444 332 544 322 444 654 665 dcb
877 655 666 877 877 766 655 544 544 755 655 322 433 554 dcb dcb
fed edc 877 544 544 666 766 766 655 655 666 554 432 a99 edc edc
edc edc edc edc cba 876 555 555 655 766 766 543 655 edc fed edd
edc edc edc edc edc edc edc aa9 a99 776 655 544 dcb edc edc fed
```
biggest terms toward 3: R4F2.12 (fine_lap>0.03_global) value 2.15 -> -0.857; R4F2.0 (fine_grad>0.1_global) value 2.68 -> -0.852; R4F2.14 (fine_lap>0.03_bottom) value 1.79 -> -0.841; R4F2.2 (fine_grad>0.1_bottom) value 2.25 -> -0.822; R4F2.37 (fine_lap>0.03_right) value 1.86 -> -0.767

true 6 (garbage truck), predicted 4 (church)
scores [0: -4.99, 1: -3.92, 2: -6.51, 3: -5.09, 4: -2.76, 5: -7.63, 6: -2.82, 7: -4.42, 8: -4.84, 9: -8.65]
pixels (the photo shrunk to 16x16 (each cell is the average of a 4x4 block): 16 rows of 16 cells separated by spaces, each cell 3 hex digits = red, green, blue, 0 (dark) to f (bright); the network itself sees the full-resolution photo):
```
ddd ccc bcc ccd fff fff ace 9cf 9cf 8bd 789 889 88a 89b aac 556
dee ded cdd ddd fff eff 8ad 8bd 8be 8ac 878 889 89a 9ac 9ab 88a
bdd dee eee eff fff eee 9ac 9ce 9ce 8ac 989 777 89a 89b 789 778
cdd bbc ccd dee dee eee 89a 79a 78a 789 777 667 89a 78a 556 556
ded ccc ccd dee bcc aaa 777 656 666 656 556 667 89a 9ac 556 789
eee ccc aab cde bcc 889 667 556 666 666 666 666 789 9ac 556 899
eee ddd abc bcc 999 778 667 556 666 656 656 666 778 667 555 899
ddc ddd abd 99a 777 667 667 556 666 555 556 556 655 656 555 677
aa9 aaa abc 99a 667 656 656 656 656 555 555 555 545 545 445 999
aaa aaa 9ab 99b 656 556 656 555 555 555 555 545 545 545 444 899
eee bbb 899 abc 788 656 667 666 777 667 677 888 667 656 445 787
eed aaa 9aa abb 888 777 666 656 666 778 666 888 888 888 788 aaa
bbb 788 bbc abc 99a 889 666 656 556 667 99a 899 555 444 444 777
abb 999 abc bcd aab 99a 778 556 667 abc dee aab 344 334 334 444
999 aa9 bcd bbc a9a 989 878 888 bcd bce cde dee 445 333 444 334
999 999 bcd aab 988 878 777 777 a9a 9ab bde cee 789 344 333 233
```
biggest terms toward 4: R4F2.11 (fine_vert_dom_bottom) value 0.83 -> -1.421; R3F2.23 (axis_aligned_bottom) value 1.43 -> -0.620; R3F1.22 (sky_blue_row0) value 0.71 -> -0.603; R3F2.11 (vert_dom_bottom) value 1.07 -> -0.583; R4F2.23 (fine_axis_aligned_bottom) value 1.09 -> -0.451

true 1 (English springer), predicted 0 (tench)
scores [0: -0.62, 1: -1.53, 2: -9.85, 3: -4.48, 4: -5.74, 5: -7.19, 6: -6.42, 7: -7.19, 8: -0.80, 9: -7.51]
pixels (the photo shrunk to 16x16 (each cell is the average of a 4x4 block): 16 rows of 16 cells separated by spaces, each cell 3 hex digits = red, green, blue, 0 (dark) to f (bright); the network itself sees the full-resolution photo):
```
9b6 9b6 9a6 8a6 895 795 795 795 795 684 685 684 584 574 573 573
796 795 795 795 695 786 bbb 877 976 aa8 685 574 584 684 584 573
795 695 685 694 684 785 ccc 767 534 876 aa8 685 584 584 584 574
795 795 685 695 695 887 a68 856 423 645 cba 685 575 574 684 684
695 786 9a7 795 684 685 987 a88 534 756 a77 897 584 584 573 584
776 756 755 987 896 795 bcb bbb 877 645 a98 786 685 574 573 573
776 877 766 767 bb8 ab7 ddd 999 989 aaa eee bcb 685 675 574 573
775 b9b 756 988 cb9 bb9 bbc bbb bbc ccc eee fff 898 676 575 685
bc7 bbb bbb ddd dcc ca8 999 aab a9a ccd ccd eee eee 786 575 574
bc7 bbb ccc dde a99 866 bab bbb bbb bcd ccd ccd dde ccc 675 675
9a5 bba abb cdd cbb 988 bbb bbb bbc ccd bcd 9aa aab ccd bcc abb
9a6 ab9 bbb bbb ddd bab bbc bbc bbb bbc aaa abc bcc ccd cde bcc
aa5 bc9 aab 9aa bbc 99a bbb aaa 99a bbc aab abc cde cde cce 777
795 9a8 685 787 bbc 666 aab 777 887 bcc 889 889 99a aab 98a 555
573 584 684 685 685 564 564 676 abb 999 565 464 464 676 787 575
684 684 584 574 573 584 574 574 675 574 564 564 574 564 565 565
```
biggest terms toward 0: R5F3.4 (dark_green_max) value 1.30 -> -1.812; R5F3.5 (light_green_max) value 1.77 -> -1.779; R5F1.8 (hedge_top_bright) value 1.34 -> -1.268; R4F3.25 (coarse_RGedge_top) value 2.31 -> -1.062; R4F3.29 (coarse_BYedge_bottom) value 1.82 -> -0.987

true 9 (parachute), predicted 8 (golf ball)
scores [0: -6.96, 1: -0.47, 2: -4.39, 3: -2.46, 4: -7.14, 5: -5.92, 6: -8.70, 7: -6.42, 8: 0.14, 9: -9.23]
pixels (the photo shrunk to 16x16 (each cell is the average of a 4x4 block): 16 rows of 16 cells separated by spaces, each cell 3 hex digits = red, green, blue, 0 (dark) to f (bright); the network itself sees the full-resolution photo):
```
788 798 999 898 899 8b7 6b3 6c4 6d4 6c3 6d3 797 899 898 9b9 ccd
788 788 788 788 788 676 566 788 788 687 797 566 788 888 aa9 9a8
788 798 688 788 899 788 555 788 677 788 677 688 788 788 9aa 787
9a8 9a8 9a8 9b8 9b8 9b8 8a7 675 765 ac8 896 ac7 ac7 ac7 ac7 797
ddc cdb cdb cdb cdb cdb ab9 454 443 565 9a8 bca bca bca bca bca
9b8 9b8 9b8 9b8 9b8 9a8 332 454 333 665 675 ac9 ab9 ab8 ab8 ab9
8b7 8b7 9b7 8b7 8b6 454 333 334 344 654 343 9b7 8b6 8b6 8b6 9b7
9b7 9c7 9c7 9c7 9c7 453 233 447 334 345 335 897 8c6 8b6 8b6 8b6
8b6 8b6 8b6 8b6 8b6 785 433 765 444 334 233 565 7a5 7b5 7b5 7b5
7a4 7b4 7b4 7b4 7b5 573 453 445 335 336 565 774 7b5 7a5 7b4 7b5
8b5 8b5 7b5 8b5 8c5 8c5 7a5 455 346 447 686 7a5 8a5 8b5 7b5 7b5
7b5 8b5 7b5 7b5 8b5 8b5 9a6 ccd 566 aab cdb 8b5 895 8b5 8b5 8b5
7b4 7b4 7b4 7b4 7b4 7b4 aa8 dde 8c6 aca cdd 7c4 6b3 7b4 7b4 7c4
7b4 7b4 7c4 7c4 7c4 7c4 aa9 bbb 8c5 9c7 dde 9d6 8d5 9d6 9d6 ad7
8c5 8c5 8d5 9d5 9d6 9d6 9a7 785 9d6 ad7 bba 9d6 9d6 9d6 9d6 9d6
9d6 9d6 9d6 9d5 8d5 8b5 684 453 8d5 8c6 555 8b5 9d5 8d5 8d5 8d5
```
biggest terms toward 8: R3F1.25 (green_row3) value 9.69 -> -3.034; R3F1.10 (green>0.02) value 6.41 -> -2.717; R1F3.6 (max_green_patch) value 0.27 -> -2.654; R5F2.11 (light_green_center) value 1.35 -> -1.787; R4F3.29 (coarse_BYedge_bottom) value 2.76 -> -1.128

true 3 (chain saw), predicted 4 (church)
scores [0: -5.47, 1: -4.69, 2: -6.41, 3: -3.46, 4: -3.08, 5: -4.41, 6: -5.82, 7: -3.67, 8: -5.24, 9: -9.24]
pixels (the photo shrunk to 16x16 (each cell is the average of a 4x4 block): 16 rows of 16 cells separated by spaces, each cell 3 hex digits = red, green, blue, 0 (dark) to f (bright); the network itself sees the full-resolution photo):
```
888 aaa bbb bbb aab 888 887 887 887 444 112 888 888 aab 555 333
777 887 888 898 888 888 887 787 887 665 111 343 777 555 433 334
777 777 887 888 888 887 887 887 887 777 222 222 565 554 553 444
777 776 887 887 888 787 777 787 887 887 555 112 555 554 554 333
777 777 887 887 887 787 776 676 776 887 555 222 889 443 663 444
777 777 887 887 887 887 776 666 565 565 333 333 455 333 777 444
776 777 777 887 887 887 887 776 555 333 111 222 667 455 333 445
777 777 887 887 887 887 887 787 776 333 221 221 221 211 322 334
777 877 777 887 887 887 887 787 777 554 443 332 211 310 211 434
776 887 777 887 887 887 787 776 776 443 554 443 321 311 222 434
766 776 777 887 776 777 777 776 776 444 432 554 433 321 223 445
666 776 777 665 433 776 776 776 776 444 554 565 454 444 111 001
666 666 776 766 411 333 665 776 776 555 775 665 555 444 223 001
666 666 776 766 411 222 665 555 666 544 332 555 454 333 222 111
666 666 776 766 422 111 222 222 111 222 222 666 555 444 111 111
555 665 766 544 322 111 111 001 000 222 333 555 555 444 111 111
```
biggest terms toward 4: R4F2.11 (fine_vert_dom_bottom) value 0.82 -> -1.645; R3F1.30 (dark_row3) value 1.73 -> -1.069; R4F2.9 (fine_vert_dom_global) value 0.82 -> -0.849; R4F2.23 (fine_axis_aligned_bottom) value 0.76 -> -0.587; R4F2.4 (fine_flat<0.05_top) value 1.01 -> -0.438

true 7 (gas pump), predicted 3 (chain saw)
scores [0: -7.47, 1: -4.20, 2: -7.61, 3: -0.81, 4: -7.13, 5: -8.91, 6: -3.75, 7: -1.94, 8: -5.18, 9: -4.21]
pixels (the photo shrunk to 16x16 (each cell is the average of a 4x4 block): 16 rows of 16 cells separated by spaces, each cell 3 hex digits = red, green, blue, 0 (dark) to f (bright); the network itself sees the full-resolution photo):
```
aa9 aaa ccb eed bbb 999 b98 a63 a74 dcc eed ddd bbc aaa 999 888
553 663 662 aa9 bbb bbb 965 a74 b75 863 aa9 ddd cdc bbb 9aa 999
553 563 452 774 997 aaa c98 b86 c86 975 998 bcc 663 886 bbb aaa
552 330 452 675 877 866 a87 a64 b86 665 764 653 553 442 775 553
441 230 450 785 ccc aaa 789 b98 cbb 89a ccb 877 cbb 443 120 120
340 220 230 553 edd dcd aab 578 579 889 cbc bba cba 655 134 133
220 220 340 552 c97 caa 567 246 345 468 c65 c33 a22 444 234 345
330 642 a64 b85 b75 c43 345 235 245 346 a42 e00 c11 665 776 665
550 651 872 542 822 c32 345 245 235 467 b64 e31 b31 887 987 776
987 aa9 877 554 865 975 256 146 246 257 976 875 876 999 aaa aaa
999 aaa 888 766 777 667 368 235 234 479 777 666 555 777 aaa aaa
a99 aaa bbb bba aa9 899 269 257 135 37a bbb cbb bbb bba aaa 999
998 aa9 bba cbb cbb aaa 334 667 866 976 bbb ccb bbb bba aa9 999
998 aaa bba ccb ccb ca9 952 cba a87 953 ccc cbb bbb bbb aaa 888
888 999 aa9 bba bbb ca8 a75 bba b98 a75 ccb cbb bbb bbb aa9 988
776 887 987 a98 ba9 ca8 a86 ba9 a97 d97 bba ba9 ba8 a98 987 776
```
biggest terms toward 3: R5F3.10 (bright_red_max) value 1.56 -> -1.118; R1F3.3 (red_blob) value 0.32 -> -0.745; R3F2.33 (grad>0.1_center) value 1.86 -> -0.715; R3F2.13 (lap>0.03_top) value 3.29 -> -0.692; R5F3.12 (bright_orange_max) value 0.62 -> -0.666

true 3 (chain saw), predicted 8 (golf ball)
scores [0: -2.19, 1: -4.31, 2: -9.83, 3: -3.54, 4: -4.70, 5: -4.30, 6: -7.31, 7: -5.30, 8: 0.20, 9: -10.06]
pixels (the photo shrunk to 16x16 (each cell is the average of a 4x4 block): 16 rows of 16 cells separated by spaces, each cell 3 hex digits = red, green, blue, 0 (dark) to f (bright); the network itself sees the full-resolution photo):
```
674 563 452 442 332 332 121 231 231 121 121 121 121 231 231 231
755 754 553 342 332 332 222 221 110 221 121 121 231 231 231 242
856 c88 552 341 342 342 231 231 351 351 341 241 231 241 352 241
866 a77 342 352 352 341 241 241 242 351 352 221 231 231 241 352
889 865 342 452 462 241 241 231 231 231 665 987 231 231 231 241
ade 689 452 452 342 231 131 121 231 241 888 baa 676 353 231 241
aee bee 686 332 221 231 131 121 121 121 766 9ac ace cde 676 231
8be adf bdd 899 876 a98 998 443 121 688 9bd ace abd 78a cde 788
69c 8bd ade acc 998 555 554 453 242 ace adf ace abe 79a ccd ccc
7bd 479 69b 8bd 7ab 566 241 341 353 8ad ace 9be 789 445 888 443
adf 9cf 9ce 578 232 221 231 563 343 79c 9bd 68a 566 666 778 342
457 567 568 344 342 231 342 442 221 8ac 78b 557 899 554 887 896
58b 58c 47a 222 343 443 453 554 432 677 68b 445 454 453 567 887
36a 58c 58b 444 775 664 ab6 895 553 674 454 322 222 345 58a 676
259 58c 68a 232 452 562 683 563 563 673 454 223 122 257 58a 454
247 36b 69c 675 673 683 672 572 573 573 574 135 344 357 68b 354
```
biggest terms toward 8: R5F2.8 (dark_green_global) value 1.80 -> -2.125; R5F3.6 (pale_blue_sky_max) value 1.99 -> -1.483; R3F2.18 (diagonal_global) value 3.17 -> -1.006; R4F3.18 (coarse_diagonal_global) value 4.90 -> -1.003; R1F3.10 (horiz_edge_bottom_bright) value 0.39 -> -0.936

true 8 (golf ball), predicted 0 (tench)
scores [0: -0.44, 1: -5.05, 2: -6.67, 3: -4.98, 4: -3.79, 5: -2.39, 6: -7.75, 7: -9.31, 8: -1.84, 9: -9.53]
pixels (the photo shrunk to 16x16 (each cell is the average of a 4x4 block): 16 rows of 16 cells separated by spaces, each cell 3 hex digits = red, green, blue, 0 (dark) to f (bright); the network itself sees the full-resolution photo):
```
311 200 311 210 100 100 100 100 100 100 000 000 000 100 100 100
521 621 a52 521 100 100 100 100 100 100 100 100 000 000 000 000
731 831 831 421 211 100 100 100 100 100 211 211 110 100 000 100
521 852 963 a73 973 532 100 100 321 521 641 753 652 542 311 101
964 b95 ca5 c96 c96 b95 210 321 431 431 763 874 984 a95 b84 942
a85 ca6 db7 db7 eb7 ec7 642 752 432 552 984 b95 a95 cb7 db7 b73
ca5 db6 db7 ec7 ec8 ec8 a74 973 752 763 a95 b95 ba6 cb7 ec8 b74
ca6 db7 ec7 ec8 ec8 ea8 ca5 a73 974 874 ca6 ca6 aa6 cb7 cb7 b84
ca6 db7 ec8 ec8 ea8 eb8 a85 a74 a84 984 a95 ba6 ba6 cb7 dc8 a63
c86 d97 da7 ea8 eb8 db8 211 642 984 b95 ca6 cb7 dc7 ed9 b96 511
c95 da7 eb8 eb8 b96 321 100 100 753 ba6 db7 ec8 ec8 ca6 531 100
521 421 432 211 100 100 100 100 100 321 643 643 852 b73 a63 311
000 100 100 100 100 100 100 100 100 100 100 100 741 b73 a63 731
000 100 100 100 100 100 100 100 100 100 100 100 421 952 963 831
000 000 100 100 100 100 100 100 100 100 100 100 100 631 842 842
000 000 000 100 100 100 100 100 100 100 100 200 100 200 631 842
```
biggest terms toward 0: R5F2.3 (tan_light_brown_center) value 1.72 -> -2.512; R5F3.13 (bright_yellow_max) value 1.95 -> -1.681; R5F3.1 (tan_light_brown_max) value 2.00 -> -1.676; R5F1.8 (hedge_top_bright) value 2.53 -> -1.632; R5F2.27 (bright_yellow_center) value 1.37 -> -1.505

true 3 (chain saw), predicted 0 (tench)
scores [0: -3.00, 1: -3.35, 2: -5.34, 3: -3.26, 4: -3.55, 5: -7.45, 6: -3.77, 7: -4.65, 8: -6.32, 9: -10.69]
pixels (the photo shrunk to 16x16 (each cell is the average of a 4x4 block): 16 rows of 16 cells separated by spaces, each cell 3 hex digits = red, green, blue, 0 (dark) to f (bright); the network itself sees the full-resolution photo):
```
dde ddd ddd eee eee ddd ddd ddd ddd ccd ccd ddd ddd ddd ccd ddd
ddd ddd ddd ddd ccc bcc ddd ddd cdd ccd ccd bcc ccc ccc ccc ccc
ccd ccc ccd ccc ddd dde ddd ddd ddd dde ddd ccc eee eee eee dde
cdd ccd cdd ddd ddd dde dde dde eee dee dde ccd dde eee dde ddd
eee cdd ccd ccc ccc cbb ccd ddd ddd ddd ddd ccc ddd dee eee eee
dde eee eee eee b97 a97 ddd ddd 999 877 cbc bbc ccc ccd ddd eee
ddd ddd ddd ddd a86 ba8 ccd ccd 888 633 623 b99 ddd ddd eee eee
ddd ddd ccc bbb 975 a85 aaa ccc bbb 523 412 733 baa aaa bbb ccc
788 555 333 333 975 874 444 444 444 422 422 422 543 332 232 232
666 665 554 564 a86 975 565 666 544 443 332 222 332 342 342 442
676 665 665 665 975 974 777 666 433 532 333 222 664 554 332 443
777 676 676 777 875 864 776 665 543 443 222 333 875 443 332 332
777 787 777 765 654 653 654 453 554 443 222 222 a86 443 343 343
776 777 776 654 654 765 553 342 452 452 221 111 764 554 443 453
976 855 a97 765 654 765 765 986 a97 a87 221 221 332 a97 765 775
665 776 776 665 665 665 765 776 665 665 554 655 665 776 665 776
```
biggest terms toward 0: R5F1.8 (hedge_top_bright) value 2.52 -> -1.356; R5F3.4 (dark_green_max) value 1.01 -> -1.343; R3F2.34 (horiz_dom_center) value 1.50 -> -0.838; R4F3.29 (coarse_BYedge_bottom) value 1.40 -> -0.788; R5F3.14 (pure_white_max) value 1.70 -> -0.773

Combine-round history (validation accuracy): round 6: 69.04%. Best so far: round 6.

Revise the final layer (or choose different feature files) to fix the remaining confusions. Submit the complete classifier again in the COMBINE format.