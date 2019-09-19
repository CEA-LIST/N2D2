#include <memory>
#include "Activation/ActivationScaling.hpp"
#include "Activation/ActivationScalingMode.hpp"

const unsigned char N2D2::DoubleShiftScaling::NO_SHIFT = std::numeric_limits<unsigned char>::max();

N2D2::ActivationScaling::ActivationScaling(): mMode(ActivationScalingMode::NONE), mScaling(nullptr) {

}

N2D2::ActivationScaling::ActivationScaling(ActivationScalingMode mode, 
                                           std::unique_ptr<AbstractScaling> scaling)
                                        : mMode(mode), mScaling(std::move(scaling))
{

}