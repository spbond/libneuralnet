/*
 * Synapse.cpp
 *
 *  Created on: Jun 14, 2014
 *      Author: shaun
 */

#include "Synapse.h"
#include <cmath>


namespace NN
{

Synapse::Synapse()
{}

Synapse::~Synapse()
{}



SigmoidSynapse::SigmoidSynapse(float coeff)
: m_coeff(coeff)
{}

float
SigmoidSynapse::evaluate(float x) const
{
    if( x < 0 )
        return 1.0 / (2.0 - m_coeff*x);
    else
        return 1.0 - 1.0 / (2.0 + m_coeff*x);
}

float
SigmoidSynapse::operator()(float x) const
{
    return evaluate(x);
}

} // End namespace NN
