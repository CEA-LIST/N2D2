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
    def __init__(self, dataset, dims, batch_size=1, compositeStimuli=False):
        self.stimuli_provider = N2D2.StimuliProvider(dataset, dims, batch_size, compositeStimuli)
        # Dictionary of transformation objects
        #self.transformations = None
        #self.dataset = dataset
        
    """def addTransformations(self, transformations)
        self.transformations = transformations
    """
    """
    def readRandomBatch(self):
        data = self.dataset.readRandomBatch()
        for trans in transformations:
            data = trans(data)
        return data
    """
    def N2D2(self):
        return self.stimuli_provider
