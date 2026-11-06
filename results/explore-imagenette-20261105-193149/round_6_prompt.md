[Round 6 of 7: COMBINE. 1 more combine round(s) with feedback after this one.]

Class numbers: 0 = tench, 1 = English springer, 2 = cassette player, 3 = chain saw, 4 = church, 5 = French horn, 6 = garbage truck, 7 = gas pump, 8 = golf ball, 9 = parachute.
Results of your classifier on the validation split:

accuracy: 70.98%

per-class accuracy:
  0:  82.6%
  1:  74.2%
  2:  70.1%
  3:  56.8%
  4:  69.0%
  5:  69.8%
  6:  73.2%
  7:  68.5%
  8:  69.5%
  9:  75.5%

confusion matrix (row = true, col = predicted):
           0     1     2     3     4     5     6     7     8     9
   0     404    16     4    14     6     9    11     3    18     4
   1       8   356     7    30    10    20     8     7    26     8
   2       9    21   372    15     7    18    36    36     8     9
   3      29    32    23   266    19    46    21    17     7     8
   4      10    12     4    19   356    35    39    19    15     7
   5      16    40    13    35    17   352     5    15     9     2
   6       9    10    32    28    32    11   368     8     3     2
   7       8    16    16    16    25    11    44   331    10     6
   8      16    31    17    22    14    18    11     7   358    21
   9       9     8     3    25    16     6    20     9    29   386

Your most frequent confusions, explained through your final layer (averaged over all validation images with that confusion; the terms plus the bias gap sum to the score gap):
```
3 (chain saw) predicted as 5 (French horn): 46 images. Average score[3] - score[5] = -1.782 (sum of neuron terms -8.659, bias gap +6.877)
  neurons pushing toward 5 (average (w[3]-w[5]) x value):
    R3F2.9 (green+1 r2c1): value 1.005, w[3] -1.018, w[5] +3.544 -> -4.587
    R3F2.10 (green+1 r2c2): value 1.003, w[3] -3.800, w[5] +0.668 -> -4.481
    R3F2.14 (green+1 r3c2 grass): value 1.005, w[3] -1.363, w[5] +1.454 -> -2.832
    R3F2.12 (green+1 r3c0 grass): value 1.001, w[3] -2.815, w[5] -0.093 -> -2.724
    R2F2.12 (fine checker texture (+) fur/grass): value 0.013, w[3] -8.527, w[5] +176.400 -> -2.423
    R2F2.13 (fine checker texture (-)): value 0.013, w[3] -25.440, w[5] +125.700 -> -1.970
  neurons pushing toward 3: R3F2.3 (green+1 r0c3) +4.624; R3F2.7 (green+1 r1c3) +2.527; R3F3.5 (fine checker energy (fur/grass/engraving)) +2.081

7 (gas pump) predicted as 6 (garbage truck): 44 images. Average score[7] - score[6] = -1.920 (sum of neuron terms -20.024, bias gap +18.104)
  neurons pushing toward 6 (average (w[7]-w[6]) x value):
    R3F2.1 (green+1 r0c1): value 0.998, w[7] -1.159, w[6] +5.593 -> -6.741
    R3F2.6 (green+1 r1c2): value 0.997, w[7] -0.162, w[6] +5.037 -> -5.184
    R3F2.2 (green+1 r0c2): value 1.006, w[7] -1.657, w[6] +3.000 -> -4.686
    R3F2.7 (green+1 r1c3): value 1.004, w[7] -1.360, w[6] +3.175 -> -4.551
    R3F3.1 (fine horizontal edge energy): value 0.364, w[7] -8.469, w[6] -0.462 -> -2.912
    R3F2.11 (green+1 r2c3): value 1.002, w[7] -1.399, w[6] +0.703 -> -2.106
  neurons pushing toward 7: R3F2.15 (green+1 r3c3 grass) +3.488; R3F2.0 (green+1 r0c0) +2.445; R3F2.9 (green+1 r2c1) +2.107

5 (French horn) predicted as 1 (English springer): 40 images. Average score[5] - score[1] = -1.678 (sum of neuron terms +2.399, bias gap -4.077)
  neurons pushing toward 1 (average (w[5]-w[1]) x value):
    R3F2.15 (green+1 r3c3 grass): value 1.007, w[5] -2.346, w[1] +1.073 -> -3.442
    R3F2.7 (green+1 r1c3): value 1.008, w[5] -2.920, w[1] +0.468 -> -3.416
    R3F2.3 (green+1 r0c3): value 1.009, w[5] -3.090, w[1] +0.155 -> -3.275
    R3F3.0 (fine vertical edge energy): value 0.344, w[5] -2.705, w[1] +4.048 -> -2.324
    R3F3.1 (fine horizontal edge energy): value 0.305, w[5] -0.391, w[1] +6.274 -> -2.033
    R4F1.7 (max bright vertical bar (pump, steeple)): value 0.490, w[5] -0.565, w[1] +3.410 -> -1.947
  neurons pushing toward 5: R3F2.9 (green+1 r2c1) +5.473; R3F2.5 (green+1 r1c1) +5.315; R3F3.2 (fine spot energy (texture)) +3.590

4 (church) predicted as 6 (garbage truck): 39 images. Average score[4] - score[6] = -1.809 (sum of neuron terms -22.656, bias gap +20.847)
  neurons pushing toward 6 (average (w[4]-w[6]) x value):
    R3F2.6 (green+1 r1c2): value 1.008, w[4] -2.816, w[6] +5.037 -> -7.913
    R3F2.7 (green+1 r1c3): value 1.010, w[4] -4.154, w[6] +3.175 -> -7.404
    R3F2.1 (green+1 r0c1): value 1.010, w[4] -0.600, w[6] +5.593 -> -6.256
    R3F2.3 (green+1 r0c3): value 1.010, w[4] -4.132, w[6] +0.731 -> -4.913
    R3F2.4 (green+1 r1c0): value 1.002, w[4] -0.996, w[6] +3.230 -> -4.233
    R3F2.2 (green+1 r0c2): value 1.012, w[4] -0.559, w[6] +3.000 -> -3.603
  neurons pushing toward 4: R3F2.15 (green+1 r3c3 grass) +6.530; R3F2.12 (green+1 r3c0 grass) +4.226; R3F2.13 (green+1 r3c1 grass) +2.734

2 (cassette player) predicted as 7 (gas pump): 36 images. Average score[2] - score[7] = -2.700 (sum of neuron terms -5.269, bias gap +2.569)
  neurons pushing toward 7 (average (w[2]-w[7]) x value):
    R3F2.5 (green+1 r1c1): value 0.990, w[2] -3.026, w[7] +1.601 -> -4.579
    R3F3.0 (fine vertical edge energy): value 0.327, w[2] -10.880, w[7] -1.160 -> -3.180
    R3F2.9 (green+1 r2c1): value 0.976, w[2] -0.901, w[7] +1.484 -> -2.328
    R3F2.4 (green+1 r1c0): value 0.991, w[2] -0.812, w[7] +1.213 -> -2.006
    R1F2.1 (vertical edges TR): value 0.459, w[2] -4.169, w[7] -0.446 -> -1.708
    R3F2.0 (green+1 r0c0): value 0.993, w[2] +0.352, w[7] +1.929 -> -1.566
  neurons pushing toward 2: R3F2.8 (green+1 r2c0) +2.961; R3F2.15 (green+1 r3c3 grass) +2.935; R3F3.1 (fine horizontal edge energy) +2.818
```

10 individual mistakes:

true 8 (golf ball), predicted 9 (parachute)
scores [0: -5.50, 1: -3.56, 2: -5.90, 3: -2.79, 4: -3.73, 5: -4.29, 6: -1.99, 7: -5.04, 8: 0.41, 9: 4.92]
pixels (the photo shrunk to 16x16 (each cell is the average of a 4x4 block): 16 rows of 16 cells separated by spaces, each cell 3 hex digits = red, green, blue, 0 (dark) to f (bright); the network itself sees the full-resolution photo):
```
89a 99a 99a 9ab 9ab 9ab aab aab aab aac aac aac aab aab aab 9ab
89a 99a 9aa 9ab 9ab aab aac aac abc abc abc abc abc abc abc aac
99a 9ab 9ab 9ab aab abc abc abc abc abc abc abc abc abc abc abc
9ab 9ab aab aac abc abc abc abc abc abc abc abc abc abc abc abc
99a 99a abc abc abc abc 99a abb ccd bcd acc acc abc abc abc abc
888 bcd cdd ddd bbc 889 abb acd 8bc 8bc cdd 9ab bcc ddd cdd bcd
89b 69b 7ac eee dde 788 ddd 8bc 9bc 7bc eee bcc eee cde 8bd 7bc
79b 69b 7bc eee dee 789 cdd cde cde cde eee cdd eee cde 9cd 7bc
9bb bcd cdd eee cdd 89a 9ab dde eee eee cdd 8aa dde eee eee dee
677 abc bdd bcd acc acc 9ab 9ab bcd bcd bcd abc 9ab cdd cdd bcd
9ab 9ab 9ac abc acc acc bcd bcd bcd bdd cdd cdd bcd abc bcd acd
9ab 9ac abc abc abc acc bcd bcd bdd cdd cdd cdd bcd bcd bcd acd
9ab 9ab aac abc abc acc bcd bcd bcd bcd cdd cdd bcd bcd bcd acd
9ab 9ab 9ab abc abc abc acc bcd bcd bcd bcd bcd bcd bcd bcd acc
89a 9ab 9ab 9ac aac abc abc acc acc bcd bcd bcd bcd acc acc abc
89a 89a 9ab 9ab aab aac abc abc abc abc acc abc abc abc abc abc
```
biggest terms toward 9: R3F2.0 (green+1 r0c0) value 0.99 -> -3.692; R1F1.0 (lum r0c0) value 0.65 -> -1.711; R1F1.3 (lum r0c3) value 0.72 -> -1.500; R1F1.10 (lum r2c2) value 0.82 -> -1.461; R1F1.13 (lum r3c1) value 0.74 -> -1.341

true 9 (parachute), predicted 0 (tench)
scores [0: -0.57, 1: -1.38, 2: -5.03, 3: -2.24, 4: -2.51, 5: -2.82, 6: -4.55, 7: -6.23, 8: -1.25, 9: -0.61]
pixels (the photo shrunk to 16x16 (each cell is the average of a 4x4 block): 16 rows of 16 cells separated by spaces, each cell 3 hex digits = red, green, blue, 0 (dark) to f (bright); the network itself sees the full-resolution photo):
```
887 887 887 887 888 888 888 888 888 898 898 898 898 898 998 998
887 887 887 897 898 898 898 898 898 998 998 998 998 998 998 998
897 997 997 998 997 997 998 998 998 998 998 998 998 998 998 998
997 997 997 997 997 998 997 998 998 997 997 998 998 998 998 998
998 998 998 998 a97 b96 997 9a8 9a8 9a8 9a8 9a8 9a8 9a8 9a8 9a8
9a8 9a8 9a8 9a8 ba7 b97 aa8 9a8 9a8 9a8 9a8 aa8 aa8 aa8 aa8 aa8
9a8 9a8 9a8 9a8 aa8 aa8 aa8 aa8 aa8 aa8 aa8 aa8 aa8 aa8 aa8 aa8
aa8 aa8 aa8 aa8 aa8 aa8 aa8 aa8 aa8 aa8 aa8 ab8 ab8 ab8 ab8 ab8
875 997 aa8 aa8 aa8 ba8 aa8 aa8 aa8 aa8 ab8 bb8 bb8 bb8 bb8 bb8
321 322 432 775 ba8 bb8 ba8 ba8 bb8 aa8 875 996 bb8 bb8 bb8 bb8
543 654 765 a97 bb8 bb8 bb8 bb8 bb8 764 432 432 ba8 bb8 bb8 bb8
aa7 986 ba7 ba7 ba7 bb8 bb7 ba7 bb8 754 431 321 bb8 bb8 bb8 bb8
543 111 764 985 985 a85 a85 a96 a96 764 421 321 985 874 874 974
431 111 643 974 a74 a74 a84 a74 a74 974 642 642 a85 974 a74 a74
542 322 532 974 974 974 974 974 974 654 544 544 865 a74 a74 974
754 543 533 864 973 863 863 864 655 554 544 555 544 544 973 863
```
biggest terms toward 0: R3F2.7 (green+1 r1c3) value 1.08 -> -12.655; R3F2.0 (green+1 r0c0) value 1.05 -> -9.467; R3F2.3 (green+1 r0c3) value 1.05 -> -9.329; R3F2.9 (green+1 r2c1) value 1.08 -> -8.648; R3F2.4 (green+1 r1c0) value 1.08 -> -6.500

true 0 (tench), predicted 5 (French horn)
scores [0: 0.17, 1: -1.40, 2: -5.63, 3: 0.43, 4: -0.11, 5: 2.25, 6: -3.99, 7: -3.43, 8: -5.30, 9: -9.65]
pixels (the photo shrunk to 16x16 (each cell is the average of a 4x4 block): 16 rows of 16 cells separated by spaces, each cell 3 hex digits = red, green, blue, 0 (dark) to f (bright); the network itself sees the full-resolution photo):
```
678 777 222 664 874 342 222 343 343 232 443 553 222 232 221 111
222 788 444 787 774 332 787 674 453 342 452 432 221 341 121 000
677 565 687 665 652 454 797 675 343 332 221 332 121 231 010 000
9aa bb9 aaa 553 873 442 553 442 653 332 453 443 110 111 110 000
986 874 432 443 873 553 643 b85 c85 a85 986 776 998 332 110 110
985 874 554 655 874 332 853 fc9 fc8 a64 bb9 665 da7 a85 111 111
bb9 cca 9aa aa9 b97 787 753 d97 eb8 da7 999 776 776 764 211 332
997 887 777 887 b97 555 432 c86 fb8 b85 555 565 444 121 221 222
554 876 bb8 887 654 555 553 a75 fb8 db7 332 442 121 111 211 111
321 863 b96 875 432 653 754 964 a75 986 ba7 dc9 332 222 211 111
442 653 543 553 764 864 975 862 963 863 a86 ec9 974 652 332 222
876 775 554 653 763 a76 c97 962 a72 b83 da7 eb7 da6 863 643 332
653 775 555 664 543 a75 b86 963 a84 b94 c96 eb8 db8 763 543 333
543 321 221 342 665 965 a64 a76 a96 ba8 887 d97 db8 332 754 632
775 552 231 452 666 977 964 ca7 db8 cb8 876 b86 ca7 986 754 643
342 342 131 231 775 a97 ba7 db9 ec9 fda a97 ca8 eda 765 433 533
```
biggest terms toward 5: R3F2.13 (green+1 r3c1 grass) value 0.99 -> -12.050; R3F2.14 (green+1 r3c2 grass) value 1.02 -> -9.344; R3F2.2 (green+1 r0c2) value 1.05 -> -6.417; R3F2.5 (green+1 r1c1) value 1.01 -> -5.446; R3F3.2 (fine spot energy (texture)) value 0.52 -> -4.166

true 8 (golf ball), predicted 5 (French horn)
scores [0: -6.36, 1: -1.09, 2: -5.61, 3: -3.64, 4: -2.43, 5: 2.79, 6: -2.75, 7: -4.50, 8: 0.50, 9: -4.24]
pixels (the photo shrunk to 16x16 (each cell is the average of a 4x4 block): 16 rows of 16 cells separated by spaces, each cell 3 hex digits = red, green, blue, 0 (dark) to f (bright); the network itself sees the full-resolution photo):
```
000 ddd fff eee bcd ddd ddd dcc cbc a99 666 111 000 000 000 000
111 eee fff fff eee eee eee eee edd edd ddd baa 333 000 000 000
111 ddd eee eee eee fff eee eed cb9 ddd ccc bbb 988 111 000 000
111 666 ccc ddd ddd fff eee cdd 689 cdd ccc bbb 888 332 000 000
111 111 554 999 ccc fff eee abc 79b ccc ddd ccd ccc bbb 888 333
111 111 111 111 99a fff eee ccc bbc dde eef ddd bb9 ccb ddd bbb
111 111 211 222 667 eee eee dde dde fff ddd bcd 898 bbc aaa cbb
111 111 332 111 222 999 ccc ccc ddd fff fff bcd 577 8ac dcd bbb
110 222 443 333 333 111 666 888 ddd fff eee ccd 79b bbb cbc ccc
000 222 221 433 332 322 111 111 ccc fff eef dde cbc dde dde eee
000 100 333 443 332 333 222 111 aaa eee eee eee ddd eee eee cde
000 000 222 333 433 221 333 221 444 ccc ddd ccc dcd fff eee eee
000 000 000 322 332 654 222 333 111 444 99a aaa ccc fff fff eef
000 000 000 000 332 443 543 222 443 111 111 334 bbb fef fff eef
000 000 000 000 000 222 322 211 221 211 111 000 999 eee eee dde
000 000 000 000 000 111 111 111 221 111 001 000 666 ddd ddd ddd
```
biggest terms toward 5: R2F2.12 (fine checker texture (+) fur/grass) value 0.02 -> -3.900; R3F2.0 (green+1 r0c0) value 1.00 -> -3.645; R3F1.15 (coarse diagonal BR) value 1.16 -> -3.565; R3F2.9 (green+1 r2c1) value 1.00 -> -3.148; R3F3.2 (fine spot energy (texture)) value 0.36 -> -2.678

true 3 (chain saw), predicted 7 (gas pump)
scores [0: -3.65, 1: -5.60, 2: -2.53, 3: 0.45, 4: -3.55, 5: -4.34, 6: -1.00, 7: 1.17, 8: -2.95, 9: -4.99]
pixels (the photo shrunk to 16x16 (each cell is the average of a 4x4 block): 16 rows of 16 cells separated by spaces, each cell 3 hex digits = red, green, blue, 0 (dark) to f (bright); the network itself sees the full-resolution photo):
```
333 999 bba bbb bbb aa9 777 888 365 576 aa8 996 253 564 875 cba
211 221 222 333 544 766 988 888 255 355 575 785 243 555 677 465
111 211 222 211 111 223 643 633 222 124 134 134 134 244 256 465
111 111 111 111 246 259 36a 469 358 36b 37b 369 37a 356 36a 999
111 111 111 245 48c 59c 48c 48c 48c 38c 37b 37a 345 344 369 aaa
111 110 445 47c 47a 58b 47b 48c 48c 37b 369 247 333 344 678 aaa
222 211 89a 69c 47b 88a 889 47b 47b 37a 369 135 233 333 988 999
211 211 678 aaa aaa b87 a76 47a 36a 47a 369 123 111 333 777 999
333 322 334 257 359 334 334 369 369 36a 369 235 111 222 665 988
322 333 333 234 124 222 334 359 36a 36a 369 246 111 111 444 988
432 533 433 333 246 332 332 358 259 369 258 247 222 111 666 988
633 633 633 434 235 222 332 357 259 258 247 136 323 322 877 988
532 533 533 433 235 233 332 357 369 369 258 246 544 532 877 988
532 533 533 532 246 245 233 247 359 259 247 123 212 533 876 887
643 643 643 643 212 113 123 124 234 113 445 112 222 422 766 877
532 532 532 644 112 112 111 112 112 112 223 112 433 322 433 766
```
biggest terms toward 7: R3F3.2 (fine spot energy (texture)) value 0.31 -> -3.369; R3F3.11 (fine red-blue spots) value 0.08 -> -3.258; R3F2.4 (green+1 r1c0) value 0.99 -> -3.098; R3F2.12 (green+1 r3c0 grass) value 0.94 -> -2.778; R3F2.0 (green+1 r0c0) value 1.00 -> -2.680

true 2 (cassette player), predicted 5 (French horn)
scores [0: -4.86, 1: -0.86, 2: -5.73, 3: 0.59, 4: -3.83, 5: 1.59, 6: -6.16, 7: -7.99, 8: -1.33, 9: 1.46]
pixels (the photo shrunk to 16x16 (each cell is the average of a 4x4 block): 16 rows of 16 cells separated by spaces, each cell 3 hex digits = red, green, blue, 0 (dark) to f (bright); the network itself sees the full-resolution photo):
```
bbb bbc cbc ccc dcd dcd ccc ccc ddd ddd ccc aaa ccc dcd ccc ccc
bbb cbc ccc dcd dcd ccd ccc dcd ddd dcd 666 444 667 dcc ccc ccc
bbc ccc ccd dcd dcd ccc dcd ddd ddd 766 544 444 999 ccc ccc ccc
ccc ccc dcd dcd ccd dcd ddd dcd 666 544 444 999 989 ccc ccc ccc
ccc ccd dcd ccd ccd ddd dcd 766 555 444 a9a 999 a99 ccc ccc ccc
ccd ccd ccd ccc ddd ccd 656 333 444 a99 999 a9a aaa ccc ccc ccc
cbc cbc bbc cbc bbc 888 323 777 a99 999 a9a 999 a9a ccc ccc ccc
bbc bbc bbc bab 777 444 544 a99 999 a9a a99 988 aaa ccc ccc ccc
bab aab aab 778 222 556 a9a 999 a99 aaa a98 999 cbc ccc cbb cbc
aab a9a 667 222 444 999 988 a99 aaa 999 a99 bbb ccc ccc cbb cbc
99a 788 667 778 999 999 999 999 a99 a99 bbb ccc ccc ccc ccc cbc
778 667 889 aaa 888 888 999 999 a99 baa ccc ccc ccc ccc ccc cbc
222 889 99a 888 777 888 888 999 baa ccc ccc ccc ccc ccc ccc cbc
333 666 777 777 989 999 999 aaa ccc ccc ccc ccc ccc ccc cbc cbc
333 555 777 888 989 999 999 cbb ccc ccc ccc ccc ccc ccc cbc bbb
333 223 777 888 888 999 bbb cbb cbb ccc ccc ccc cbc cbc bbb bbb
```
biggest terms toward 5: R3F2.13 (green+1 r3c1 grass) value 0.99 -> -5.858; R3F2.5 (green+1 r1c1) value 0.98 -> -5.432; R3F3.5 (fine checker energy (fur/grass/engraving)) value 0.04 -> -4.973; R3F2.9 (green+1 r2c1) value 0.99 -> -4.399; R3F2.10 (green+1 r2c2) value 0.99 -> -2.995

true 1 (English springer), predicted 4 (church)
scores [0: -4.88, 1: -0.91, 2: -4.45, 3: -2.92, 4: 0.84, 5: -2.60, 6: -3.04, 7: -2.87, 8: -1.02, 9: -5.38]
pixels (the photo shrunk to 16x16 (each cell is the average of a 4x4 block): 16 rows of 16 cells separated by spaces, each cell 3 hex digits = red, green, blue, 0 (dark) to f (bright); the network itself sees the full-resolution photo):
```
9bd 8ab 778 788 555 8ac 9ce 9ce 9ce 9ce 9ce 9ce 9ce 9ce 9ce 9ce
655 432 321 532 322 333 8bc ace ace ace ace ace ace ace ace ace
432 211 321 432 321 211 789 ade ade ade ade ade ade ade ace ace
321 321 432 532 532 322 556 ade ade ade ade ade ade ade ade ade
321 321 753 853 642 321 432 9bc ade ade ade ade ade ade ade ade
432 421 864 854 642 422 432 789 ace ace ace acd bcd bdd acd acd
766 321 753 854 532 532 754 754 9bc acd acd 9bd abd 9bd 9bd 9bd
998 332 542 753 532 642 643 643 678 78a 78a 68a 79a 78a 789 689
cbb 655 643 643 643 643 542 643 543 678 578 577 567 567 567 456
bbb 876 753 743 532 532 432 642 532 565 466 576 566 566 676 666
eee 987 643 532 643 642 322 753 321 444 565 565 564 675 675 664
ddd aa9 543 864 753 753 432 642 322 664 674 574 674 674 674 674
eed bba 654 864 753 532 421 211 332 674 674 674 774 785 785 774
ddc cba 875 753 643 432 322 553 774 674 885 885 885 785 785 885
ddd ccb a97 653 432 422 664 785 774 785 775 885 785 785 774 674
788 9aa aaa 554 343 454 674 785 785 785 785 785 674 774 674 674
```
biggest terms toward 4: R3F2.12 (green+1 r3c0 grass) value 1.01 -> -3.107; R3F2.5 (green+1 r1c1) value 0.98 -> -3.070; R3F3.2 (fine spot energy (texture)) value 0.21 -> -2.920; R3F2.11 (green+1 r2c3) value 1.08 -> -2.355; R1F2.14 (antidiag edges BL) value 0.42 -> -1.460

true 9 (parachute), predicted 4 (church)
scores [0: -6.35, 1: -2.29, 2: -6.08, 3: -2.43, 4: 0.88, 5: -0.74, 6: -5.92, 7: -3.64, 8: -0.30, 9: -0.48]
pixels (the photo shrunk to 16x16 (each cell is the average of a 4x4 block): 16 rows of 16 cells separated by spaces, each cell 3 hex digits = red, green, blue, 0 (dark) to f (bright); the network itself sees the full-resolution photo):
```
ba9 ba9 ba9 ba9 ba9 ba9 ba9 aa8 aa8 ba9 ba9 bb9 bb9 bb9 bb9 bb9
ba9 ba9 ba9 ba9 ba9 ba9 ba9 654 765 bb9 bb9 bb9 bb9 bb9 bb9 bb9
ba9 ba9 ba9 ba9 ba9 bb9 bb9 876 997 bba cba cba bba bb9 bb9 bb9
ba9 ba9 ba9 ba9 bb9 bb9 bb9 ba9 ba9 cba cba cba cba bba bba bba
ba9 ba9 bb9 bb9 bb9 bb9 bba bba cba cba cba cba bba bba bba cba
bb9 bb9 bb9 bb9 bb9 bb9 bba cba bba bb9 bb9 bb9 bb9 bb9 bb9 bba
ba9 ba9 bb9 bb9 bb9 bb9 bb9 bb9 ba9 bb9 bb9 bb9 bb9 bb9 bb9 bb9
ba9 ba9 ba9 ba9 ba9 ba9 bb9 bb9 aa8 ba9 ba9 ba9 ba9 bb9 bb9 ba9
ba8 ba8 aa8 ba8 a98 ba9 ba9 ba8 a97 aa8 ba8 ba9 ba9 ba9 ba9 ba9
aa8 876 a98 ba8 a98 a97 ba9 aa8 987 987 aa8 aa8 ba9 aa8 a98 ba8
a97 987 876 a98 997 a98 997 ba8 987 987 a98 997 ba9 a98 987 a97
886 a97 876 a98 987 a97 ba8 997 987 a98 886 997 ba8 886 876 a98
886 886 665 a97 986 a98 aa8 aa8 886 a98 987 886 a98 765 665 987
765 765 654 a98 886 a98 a97 a98 886 aa8 776 776 987 765 544 886
553 543 554 987 876 a98 997 997 886 a98 876 543 886 987 433 886
543 443 765 876 876 997 997 876 876 997 876 443 876 987 332 886
```
biggest terms toward 4: R3F2.12 (green+1 r3c0 grass) value 1.02 -> -4.203; R3F2.9 (green+1 r2c1) value 1.03 -> -3.241; R3F3.2 (fine spot energy (texture)) value 0.24 -> -2.465; R3F2.15 (green+1 r3c3 grass) value 1.02 -> -1.766; R3F3.0 (fine vertical edge energy) value 0.23 -> -1.759

true 9 (parachute), predicted 0 (tench)
scores [0: 3.30, 1: -0.84, 2: -5.22, 3: -2.34, 4: -3.70, 5: -3.35, 6: -4.03, 7: -4.67, 8: 2.05, 9: -8.11]
pixels (the photo shrunk to 16x16 (each cell is the average of a 4x4 block): 16 rows of 16 cells separated by spaces, each cell 3 hex digits = red, green, blue, 0 (dark) to f (bright); the network itself sees the full-resolution photo):
```
332 332 332 443 533 443 554 332 342 675 564 453 453 332 443 332
433 432 432 332 533 544 565 222 333 342 342 343 232 232 222 121
333 332 322 443 655 332 332 342 342 342 342 342 232 222 121 111
673 674 453 888 ccc 684 684 684 673 573 573 462 462 342 342 242
463 463 573 898 ccb 574 452 352 352 352 352 676 564 242 242 232
563 573 573 ccb ddd 9a8 784 684 683 563 332 444 678 453 352 342
784 563 442 786 895 794 793 793 793 793 8a5 9a8 463 563 462 452
673 683 783 783 783 783 783 794 794 794 8a6 684 452 352 352 342
573 683 784 683 673 683 683 683 784 9a7 895 783 673 673 572 572
894 793 894 894 783 683 894 9a5 ab7 9a6 9a6 8a4 894 783 573 793
894 894 894 783 784 784 885 895 895 9a7 895 894 794 894 673 572
673 773 673 673 563 564 563 462 564 563 563 452 452 452 452 452
563 462 462 563 463 453 443 453 553 573 573 573 462 463 562 462
884 894 784 553 222 222 122 234 333 573 563 663 563 563 452 452
452 452 352 221 111 111 112 122 222 442 342 352 452 352 342 342
453 562 452 232 121 111 112 111 222 452 452 342 342 452 452 452
```
biggest terms toward 0: R3F2.7 (green+1 r1c3) value 1.12 -> -13.096; R3F2.3 (green+1 r0c3) value 1.07 -> -9.504; R3F2.9 (green+1 r2c1) value 1.18 -> -9.408; R3F2.0 (green+1 r0c0) value 1.03 -> -9.335; R3F2.4 (green+1 r1c0) value 1.15 -> -6.936

true 1 (English springer), predicted 5 (French horn)
scores [0: -3.60, 1: 0.81, 2: -4.33, 3: 1.21, 4: -5.89, 5: 7.16, 6: -6.82, 7: -6.38, 8: -3.13, 9: -5.81]
pixels (the photo shrunk to 16x16 (each cell is the average of a 4x4 block): 16 rows of 16 cells separated by spaces, each cell 3 hex digits = red, green, blue, 0 (dark) to f (bright); the network itself sees the full-resolution photo):
```
ca9 cba dcb cbb cbb dcb cbb baa b99 987 b98 dbb dba cba cba cba
cba cba cbb bbb baa bab 665 433 544 544 877 987 cba dcb dcb cba
977 cba cba 877 778 333 433 645 656 655 433 877 a98 b98 dcb dcb
ba9 987 b99 cbb 444 322 433 655 756 544 433 323 988 ba9 a88 cba
cba cba a98 999 223 322 644 766 867 644 433 222 544 cbb cbb a99
bba cba cba ccc 333 322 755 ccc a99 644 322 222 433 cba baa cbb
baa baa baa 999 434 433 999 a88 a9a 544 433 556 988 cbb cbb ccb
cba baa ba9 988 555 766 999 644 988 655 544 555 baa cbb ccb dcb
cba cba b99 bbb bbb baa a9a 988 888 755 888 888 cba cbb cbb cbb
cba cba b99 ccc ddd ccc cbc bbb bab bbb bbb dde dcc cbb cbb cbb
cba cba a99 ccc ddd ccd ccc bbb bbb cbc ccd dde ddd cba cbb cbb
877 ba9 ba9 bbb ddd ccd bbb bab baa bbb ccc ddd cbb 976 caa dbb
cba 876 a88 988 ccc bbb ccc cbc baa baa cbc ccc cbb a99 a98 a98
cba cba 877 887 ccc a99 655 a99 988 765 999 ccc ddd bba cbb cba
caa caa ba9 988 888 999 654 655 655 544 766 aaa bbb ccb ccb dcb
aaa baa b99 a98 887 655 655 755 765 765 776 877 baa ccc dcc dcc
```
biggest terms toward 5: R2F2.12 (fine checker texture (+) fur/grass) value 0.03 -> -6.547; R3F2.9 (green+1 r2c1) value 0.97 -> -5.279; R3F3.2 (fine spot energy (texture)) value 0.48 -> -5.198; R3F2.5 (green+1 r1c1) value 0.96 -> -5.115; R2F2.13 (fine checker texture (-)) value 0.03 -> -4.764

Combine-round history (validation accuracy): round 5: 70.98%. Best so far: round 5.

Revise the final layer (or choose different feature files) to fix the remaining confusions. Submit the complete classifier again in the COMBINE format.