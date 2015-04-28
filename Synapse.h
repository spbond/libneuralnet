/*
 * Synapse.h
 *
 *  Created on: Jun 14, 2014
 *      Author: shaun
 */

#ifndef SYNAPSE_H_
#define SYNAPSE_H_

namespace NN
{

class Synapse
{
public:
    Synapse();
    virtual ~Synapse();

    virtual float
    evaluate(float) const = 0;

    virtual float
    operator()(float) const = 0;
};

class SigmoidSynapse : public Synapse
{
public:
    SigmoidSynapse(float coeff = 1);

    virtual float
    evaluate(float) const;

    virtual float
    operator()(float) const;

private:
    float m_coeff;
};

} /* namespace NN */
#endif /* SYNAPSE_H_ */
