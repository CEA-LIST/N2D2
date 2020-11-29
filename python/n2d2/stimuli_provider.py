"""
    (C) Copyright 2020 CEA LIST. All Rights Reserved.
    Contributor(s): Cyril MOINEAU (cyril.moineau@cea.fr) 
                    Johannes THIELE (johannes.thiele@cea.fr)
                    Olivier BICHLER (olivier.bichler@cea.fr)

    This software is governed by the CeCILL-C license under French law and
    abiding by the rules of distribution of free software.  You can  use,
    modify and/ or redistribute the software under the terms of the CeCILL-C
    license as circulated by CEA, CNRS and INRIA at the following URL
    "http://www.cecill.info".

    As a counterpart to the access to the source code and  rights to copy,
    modify and redistribute granted by the license, users are provided only
    with a limited warranty  and the software's author,  the holder of the
    economic rights,  and the successive licensors  have only  limited
    liability.

    The fact that you are presently reading this means that you have had
    knowledge of the CeCILL-C license and that you accept its terms.
"""

import N2D2

class StimuliProvider():
    # Be careful to match default parameters in python and N2D2 constructor
    def __init__(self, database, Size, BatchSize=1, CompositeStimuli=False):
        self._database = database
        self._Size = Size
        self._BatchSize = BatchSize
        self._CompositeStimuli = CompositeStimuli

        self._stimuli_provider = N2D2.StimuliProvider(database=self._database.N2D2(),
                                                      size=self._Size,
                                                      batchSize=self._BatchSize,
                                                      compositeStimuli=self._CompositeStimuli)

        # Dictionary of transformation objects
        #self.transformations = None
        
    """def addTransformations(self, transformations)
        self.transformations = transformations
    """

    def readRandomBatch(self, set):
        return self._stimuli_provider.readRandomBatch(set=self._database.StimuliSets['Learn'])

    def N2D2(self):
        return self._stimuli_provider
