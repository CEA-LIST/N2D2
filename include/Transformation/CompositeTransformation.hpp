/*
    (C) Copyright 2016 CEA LIST. All Rights Reserved.
    Contributor(s): Olivier BICHLER (olivier.bichler@cea.fr)

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

#ifndef N2D2_COMPOSITETRANSFORMATION_H
#define N2D2_COMPOSITETRANSFORMATION_H

#include "Transformation.hpp"
#include "utils/Utils.hpp"

namespace N2D2 {
class CompositeTransformation : public Transformation {
public:
    using Transformation::apply;

    CompositeTransformation() {};
    /// Any transformation can be transformed to a composite transformation
    template <class T> CompositeTransformation(const T& transformation);
    template <class T>
    CompositeTransformation(const std::shared_ptr<T>& transformation);
    inline void apply(cv::Mat& frame,
                      cv::Mat& labels,
                      std::vector<std::shared_ptr<ROI> >& labelsROI,
                      int /*id*/ = -1);
    inline void reverse(cv::Mat& frame,
                        cv::Mat& labels,
                        std::vector<std::shared_ptr<ROI> >& labelsROI,
                        int /*id*/ = -1);
    template <class T> void push_back(const T& transformation);
    template <class T> void push_back(const std::shared_ptr<T>& transformation);
    inline void push_back(const CompositeTransformation& transformation);
    inline bool empty() const;
    inline unsigned int size() const;
    inline std::shared_ptr<Transformation> operator[](unsigned int k);
    inline const std::shared_ptr<Transformation>
    operator[](unsigned int k) const;
    std::shared_ptr<CompositeTransformation> clone() const
    {
        return std::shared_ptr<CompositeTransformation>(doClone());
    }
    virtual ~CompositeTransformation() {};

private:
    inline virtual CompositeTransformation* doClone() const;

    std::vector<std::shared_ptr<Transformation> > mTransformationSet;
};
}

template <class T>
N2D2::CompositeTransformation::CompositeTransformation(const T& transformation)
{
    mTransformationSet.push_back(std::make_shared<T>(transformation));
}

template <class T>
N2D2::CompositeTransformation::CompositeTransformation(const std::shared_ptr
                                                       <T>& transformation)
{
    mTransformationSet.push_back(transformation);
}

void N2D2::CompositeTransformation::apply(cv::Mat& frame,
                                          cv::Mat& labels,
                                          std::vector
                                          <std::shared_ptr<ROI> >& labelsROI,
                                          int id)
{
    for (std::vector<std::shared_ptr<Transformation> >::const_iterator it
         = mTransformationSet.begin(),
         itEnd = mTransformationSet.end();
         it != itEnd;
         ++it) {
        (*it)->apply(frame, labels, labelsROI, id);
    }
}

void N2D2::CompositeTransformation::reverse(cv::Mat& frame,
                                            cv::Mat& labels,
                                            std::vector
                                            <std::shared_ptr<ROI> >& labelsROI,
                                            int id)
{
    std::vector<cv::Mat> frameSteps(1, frame.clone());

    // Forward
    for (int i = 0, size = (int)mTransformationSet.size() - 1; i < size; ++i) {
        frameSteps.push_back(frameSteps.back().clone());
        mTransformationSet[i]->apply(frameSteps.back(), id);
    }

    // Reverse
    for (int i = mTransformationSet.size() - 1; i >= 0; --i) {
        mTransformationSet[i]
            ->reverse(frameSteps.back(), labels, labelsROI, id);
        frameSteps.pop_back();
    }
}

template <class T>
void N2D2::CompositeTransformation::push_back(const T& transformation)
{
    mTransformationSet.push_back(std::make_shared<T>(transformation));
}

template <class T>
void N2D2::CompositeTransformation::push_back(const std::shared_ptr
                                              <T>& transformation)
{
    mTransformationSet.push_back(transformation);
}

void N2D2::CompositeTransformation::push_back(const CompositeTransformation
                                              & transformation)
{
    mTransformationSet.insert(mTransformationSet.end(),
                              transformation.mTransformationSet.begin(),
                              transformation.mTransformationSet.end());
}

bool N2D2::CompositeTransformation::empty() const
{
    return mTransformationSet.empty();
}

unsigned int N2D2::CompositeTransformation::size() const
{
    return mTransformationSet.size();
}

inline std::shared_ptr<N2D2::Transformation> N2D2::CompositeTransformation::
operator[](unsigned int k)
{
    return mTransformationSet.at(k);
}

inline const std::shared_ptr<N2D2::Transformation>
N2D2::CompositeTransformation::operator[](unsigned int k) const
{
    return mTransformationSet.at(k);
}

N2D2::CompositeTransformation* N2D2::CompositeTransformation::doClone() const
{
    CompositeTransformation* newTrans = new CompositeTransformation();

    for (std::vector<std::shared_ptr<Transformation> >::const_iterator it
         = mTransformationSet.begin(),
         itEnd = mTransformationSet.end();
         it != itEnd;
         ++it) {
        newTrans->push_back((*it)->clone());
    }

    return newTrans;
}

#endif // N2D2_COMPOSITETRANSFORMATION_H
