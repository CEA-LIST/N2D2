#include <memory>
#include "Activation/ActivationScaling.hpp"
#include "Activation/ActivationScalingMode.hpp"


N2D2::ActivationScaling::ActivationScaling(): mMode(ActivationScalingMode::NONE), mScaling(nullptr) {

}

N2D2::ActivationScaling::ActivationScaling(ActivationScalingMode mode, 
                                           std::unique_ptr<AbstractScaling> scaling)
                                        : mMode(mode), mScaling(std::move(scaling))
{

}