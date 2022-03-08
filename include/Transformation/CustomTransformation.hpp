/*
    (C) Copyright 2020 CEA LIST. All Rights Reserved.
    Contributor(s): Cyril MOINEAU (cyril.moineau@cea.fr)

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
*/
#include "Transformation.hpp"

namespace N2D2 {
/*  
*  CustomTransformation is a class added specially for the python API n2d2.
*  This class is here to be overide on the ptyhon side.
*  We can't just override the virtual Transformation class for the following reason :
*  Transformation are applied with the stimuliProvider by converting implicitly Transformations to CompositeTransformation.
*  But pybind doesn't allow to convert implicitly a virtual class. We thus have to ask pybind to implicitly convert every class.
*  
*  If we want to create CustomTransformation in python we have to create a non virtual object on the cpp side this class being implicitly convertible to a CompositeTransformation.
*  Then we can safely override the apply method on the python side. 
*/
class CustomTransformation : public Transformation {
public:
    CustomTransformation();
    CustomTransformation(const CustomTransformation&);
    using Transformation::apply;

    void apply(cv::Mat& frame,
               cv::Mat& /*labels*/,
               std::vector<std::shared_ptr<ROI> >& /*labelsROI*/,
               int /*id*/ = -1);

    /*
    We don't use template because Pybind need the C++ and python class to have the same name in order
    to be able to override it. Further more virtual classes can't be templated.
    */

    virtual void apply_int(
        Tensor<int>& /*frame*/,
        Tensor<int>& /*labels*/,
        std::vector<std::shared_ptr<ROI> >& /*labelsROI*/,
        int /*id*/)
        {
            std::cout << "int" << std::endl;
        };
    virtual void apply_float(
        Tensor<float>& /*frame*/,
        Tensor<int>& /*labels*/,
        std::vector<std::shared_ptr<ROI> >& /*labelsROI*/,
        int /*id*/)
        {
            std::cout << "float" << std::endl;
        };  
    virtual void apply_double(
        Tensor<double>& /*frame*/,
        Tensor<int>& /*labels*/,
        std::vector<std::shared_ptr<ROI> >& /*labelsROI*/,
        int /*id*/)
        {
            std::cout << "double" << std::endl;
        };
    virtual void apply_unsigned_char(
        Tensor<unsigned char>& /*frame*/,
        Tensor<int>& /*labels*/,
        std::vector<std::shared_ptr<ROI> >& /*labelsROI*/,
        int /*id*/)
        {
            std::cout << "uchar" << std::endl;
        };
    virtual void apply_char(
        Tensor<char>& /*frame*/,
        Tensor<int>& /*labels*/,
        std::vector<std::shared_ptr<ROI> >& /*labelsROI*/,
        int /*id*/)
        {
            std::cout << "char" << std::endl;
        };
    virtual void apply_unsigned_short(
        Tensor<unsigned short>& /*frame*/,
        Tensor<int>& /*labels*/,
        std::vector<std::shared_ptr<ROI> >& /*labelsROI*/,
        int /*id*/)
        {
            std::cout << "ushort" << std::endl;
        };
    virtual void apply_short(
        Tensor<short>& /*frame*/,
        Tensor<int>& /*labels*/,
        std::vector<std::shared_ptr<ROI> >& /*labelsROI*/,
        int /*id*/)
        {
            std::cout << "short" << std::endl;
        };


    static const char* Type;
    const char* getType() const
    {
        return Type;
    };
    int getOutputsDepth(int depth) const
    {
        return depth;
    };
    virtual ~CustomTransformation() {};

private:
    virtual CustomTransformation* doClone() const
    {
        return new CustomTransformation(*this);
    }
};
}
