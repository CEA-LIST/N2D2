#!/usr/bin/gnuplot
wrap(str,maxLength)=(strlen(str)<=maxLength)?str:str[0:maxLength]."\n".wrap(str[maxLength+1:],maxLength)
set key outside
set style histogram cluster gap 3
set style data histograms
set style fill pattern 1.00 border
set mytics 10
set grid
set tmargin 4
set bmargin 3
set xtics rotate
set boxwidth 1.0
if (!exists("multiplot")) set term png size 4550,600 enhanced
if (!exists("multiplot")) set output "Score_0/ConfusionMatrix_vis_GTSRB_score.png" 
plot "Score_0/ConfusionMatrix_vis_GTSRB_score.dat" i 0 using 3:xticlabels(wrap(stringcolumn(2),3)) ti col, '' i 0 using 5 ti col, '' i 0 using 7 ti col, '' i 0 using 8 ti col
