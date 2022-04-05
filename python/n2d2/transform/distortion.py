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

from n2d2.n2d2_interface import ConventionConverter
from n2d2.transform.transformation import Transformation
from n2d2.utils import inherit_init_docstring

import N2D2

@inherit_init_docstring()
class Distortion(Transformation):
    """
    Apply elastic distortion to the image. 
    This transformation is generally used on-the-fly 
    (so that a different distortion is performed for each image), and for the learning only.
    """

    _Type = "Distortion"
    _parameters={
        "elastic_gaussian_size": "ElasticGaussianSize",
        "elastic_sigma": "ElasticSigma",
        "elastic_scaling": "ElasticScaling",
        "scaling": "Scaling",
        "rotation": "Rotation",
        "ignore_missing_data": "IgnoreMissingData",
    }
    _convention_converter= ConventionConverter(_parameters)

    def __init__(self, **config_parameters):
        """
        :param elastic_gaussian_size: Size of the gaussian for elastic distortion (in pixels), default=15
        :type elastic_gaussian_size: int, optional
        :param elastic_sigma: Sigma of the gaussian for elastic distortion, default=6.0
        :type elastic_sigma: float, optional
        :param elastic_scaling: Scaling of the gaussian for elastic distortion, default=0.0
        :type elastic_scaling: float, optional
        :param scaling: Maximum random scaling amplitude (+/-, in percentage), default=0.0
        :type scaling: float, optional
        :param rotation: Maximum random rotation amplitude (+/-, in °), default=0.0
        :type rotation: float, optional
        """
        Transformation.__init__(self, **config_parameters)

        self._N2D2_object = N2D2.DistortionTransformation()

        # Scaling parameter is written with an upper case here but with a lower case in other classes
        # Treating this as an exception
        # if "scaling" in self._config_parameters: 
        #     self._N2D2_object.setParameter("Scaling", str(self._config_parameters.pop('scaling')))

        self._set_N2D2_parameters(self._config_parameters)
        self.load_N2D2_parameters(self.N2D2())
