#include <memory>
#include "Scaling.hpp"
#include "ScalingMode.hpp"


N2D2::Scaling::Scaling(): mMode(ScalingMode::NONE), mScaling(nullptr) {

}

N2D2::Scaling::Scaling(ScalingMode mode, std::unique_ptr<AbstractScaling> scaling)
                                        : mMode(mode), mScaling(std::move(scaling))
{

}