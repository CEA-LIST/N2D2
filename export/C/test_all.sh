#!/bin/sh
################################################################################
# Author: Olivier BICHLER (olivier.bichler@cea.fr)
# (C) Copyright 2014 CEA LIST
################################################################################

success=0
total=0
stimuli=$(ls stimuli -1 | wc -l)

for f in stimuli/*
do
    output="$($1 $f)"

    if [ "$output" = "SUCCESS" ]; then
        success=$((success + 1))
    fi

    total=$((total + 1))
    echo "$success/$total    (avg = $(echo "scale=2; 100.0*$success/$total" | bc -l)%," \
        "max = $(echo "scale=2; 100.0*($stimuli - ($total - $success))/$stimuli" | bc -l)%)"
done

echo "Tested $total stimuli"
echo "Success rate = $(echo "100.0*$success/$total" | bc -l)"
