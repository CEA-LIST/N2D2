#!/bin/sh
################################################################################
#    (C) Copyright 2014 CEA LIST. All Rights Reserved.
#    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)
#
#    This software is governed by the CeCILL-C license under French law and
#    abiding by the rules of distribution of free software.  You can  use,
#    modify and/ or redistribute the software under the terms of the CeCILL-C
#    license as circulated by CEA, CNRS and INRIA at the following URL
#    "http://www.cecill.info".
#
#    As a counterpart to the access to the source code and  rights to copy,
#    modify and redistribute granted by the license, users are provided only
#    with a limited warranty  and the software's author,  the holder of the
#    economic rights,  and the successive licensors  have only  limited
#    liability.
#
#    The fact that you are presently reading this means that you have had
#    knowledge of the CeCILL-C license and that you accept its terms.
################################################################################

DIR=`date '+%y.%m.%d-%Hh%M'`_`basename $1`

# Copy files to a local directory
mkdir $DIR
cp n2d2 n2d2.* $DIR/
chmod -w $DIR/n2d2 $DIR/n2d2.*
INI_FILE=`basename $1`
ORG_INI_FILE=$1
cp ${ORG_INI_FILE} $DIR/${INI_FILE}
chmod -w $DIR/${INI_FILE}
shift # drop $1 argument

# Simulation scripts
LEARN_ARGS=$(echo $@)
printf "#!/bin/sh\n"\
"if [ -d \"weights_init\" ]; then\n"\
"    read -p \"Previous learning results will be overridden, continue (y/n)? \" RET\n"\
"    YES=\`echo \$RET | grep -Ei '^(y|yes)\$'\`\n"\
"    if [ ! \$YES ]; then\n"\
"        exit 1\n"\
"    fi\n"\
"fi\n"\
"rm -rf _cache\n"\
"./n2d2 \"${INI_FILE}\" ${LEARN_ARGS} \$@ 2>&1 | tee learn_output.log\n" > $DIR/learn.sh
chmod +x $DIR/learn.sh

TEST_ARGS=$(echo $@ | sed 's/-learn-stdp\s\+[0-9]\+\b//g' | sed 's/-learn\s\+[0-9]\+\b/-test/g')
printf "#!/bin/sh\n"\
"if [ -d \"weights\" ]; then\n"\
"    ln -s weights/*.syntxt .\n"\
"fi\n"\
"if [ -d \"weights_stdp\" ]; then\n"\
"    ln -s weights_stdp/*.syntxt .\n"\
"fi\n"\
"./n2d2 \"${INI_FILE}\" ${TEST_ARGS} \$@ 2>&1 | tee test_output.log\n" > $DIR/test.sh
chmod +x $DIR/test.sh

printf "#!/bin/sh\n"\
"rm -rf *.syntxt *.log\n" > $DIR/clean.sh
chmod +x $DIR/clean.sh

printf "#!/bin/sh\n"\
"NOW=\`date +'.before_%%y.%%m.%%d-%%Hh%%Ms%%S'\`\n"\
"mkdir \$NOW\n"\
"mv n2d2 n2d2.* ${INI_FILE} \$NOW/\n"\
"cp ../n2d2 ../n2d2.* ${ORG_INI_FILE} .\n"\
"chmod -w n2d2 n2d2.* ${INI_FILE}\n"\
"echo \"\$NOW\" >> history.log" > $DIR/update.sh
chmod +x $DIR/update.sh

printf "#!/bin/sh\n"\
"for fn in weights/*.syntxt; do\n"\
"    awk '{ n=split(\$0, num, \" \"); for (i=1;i<=n;i++) printf(\"%%f \", (num[i]+1.0)/2.0); printf(\"\\\\n\"); }'"\
" \$fn > \$(basename \$fn)\n"\
"done\n" > $DIR/unity_wrange.sh
chmod +x $DIR/unity_wrange.sh

# Run the simulation
cd $DIR
./learn.sh
