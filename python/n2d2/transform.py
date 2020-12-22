"""
    (C) Copyright 2020 CEA LIST. All Rights Reserved.
    Contributor(s): Cyril MOINEAU (cyril.moineau@cea.fr) 
                    Johannes THIELE (johannes.thiele@cea.fr)

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


"""
For the tranformations, it is relatively important to be able to write custom 
transformations in Python. This is because these transformations are anyways not
executed on GPU at the moment, and additionaly they depend strongly on the data
that is used.
However, at the moment this remains tricky due to how data is loaded in an optimized
way in N2D2
"""

"""
Example for custom transformation
"""
"""
class CustomRescaleTransform():
    def __init__(self, factor):
        self.factor = factor
        
    def __call__(self, x):
        return self.factor*x
"""
"""
Example for a Python wrapper around the binding
"""
"""
class PadCropTransformation():
    def __init__(self, dimX, dimY):
        self.trans = N2D2.PadCropTransformation(dimX, dimY)
        
    def __call__(self, x):
        return self.trans.apply(x)
"""
class Transformation():
    def __init__(self, transformations):
        if not isinstance(transformations, list):
            raise TypeError("Wanted ", type(list), " got ", type(transformations))
        if not transformations:
            raise ValueError("Parameter transformations must not be empty")
        self._transformation = N2D2.CompositeTransformation(transformations[0].N2D2())
        for transformation in transformations[1:]:
            self._transformation.push_back(transformation.N2D2())
        
    def N2D2(self):
        return self._transformation

class PadCropTransformation(Transformation):
    def __init__(self, dimX, dimY):
        self._transformation = N2D2.PadCropTransformation(dimX, dimY)

class DistortionTransformation(Transformation):
    def __init__(self, ElasticGaussianSize=0, ElasticSigma=0, ElasticScaling=0,Scaling=0, Rotation=0):
        self._transformation = N2D2.DistortionTransformation()
        self._transformation.setParameter("ElasticGaussianSize", str(ElasticGaussianSize))
        self._transformation.setParameter("ElasticSigma", str(ElasticSigma))
        self._transformation.setParameter("ElasticScaling", str(ElasticScaling))
        self._transformation.setParameter("Scaling", str(Scaling))
        self._transformation.setParameter("Rotation", str(Rotation))