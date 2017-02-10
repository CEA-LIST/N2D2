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

DIR="$( cd "$( dirname "$0" )" && pwd )"
SELF="$( basename "$0" )"

echo "Running unit tests..."
cd ${DIR}

rc=0

for f in $(find . -maxdepth 2 -type f);
do
    # Check for ELF file for Windows 10 Ubuntu, as -x check doesn't work
    if [ -x "$f" ] && [ "$f" != "$SELF" ] && file "$f" | grep -q "ELF"; then
        x_last_modified=`stat -c "%Y" $f`
        if [ -f "$f.log" ]; then
            log_last_modified=`stat -c "%Y" $f.log`
        fi

        if [ ! -f "$f.log" ] || [ -f "$f.error" ] \
          || [ "$x_last_modified" -gt "$log_last_modified" ]; then
            log=$(./${f} 2>&1)
            rc=$?
            echo "$log" | tee ${f}.log
            echo "--------"
            if [ "$rc" != 0 ]; then
                touch ${f}.error
                break
            else
                rm -f ${f}.error
            fi
        else
            cat ${f}.log
            if [ -f "${f}.error" ]; then
                rc=-1
                break
            fi
        fi
    fi
done

if [ "$rc" = 0 ]; then
    echo "$(tput setaf 2)All tests ran successfully$(tput sgr0)"
else
    echo "$(tput setaf 1)!!! Error(s) occured during testing !!!$(tput sgr0)"

    exit $rc
fi
