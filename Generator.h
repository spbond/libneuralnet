/*
 * Generator.h
 *
 *  Created on: Jul 2, 2014
 *      Author: shaun
 */

#ifndef GENERATOR_H_
#define GENERATOR_H_

#include "NeuralNet.h"
using NN::NeuralNet;

#include <vector>
using std::vector;

#include <memory>
using std::shared_ptr;

using std::size_t;



namespace NN
{

class Generator
{
public:
    Generator(size_t population_size, vector<float> std_devs);
    virtual ~Generator();

    vector<shared_ptr<NeuralNet>>
    generate(vector<shared_ptr<NeuralNet>>);

private:
    size_t m_pop_size;
    vector<float> m_std_devs;
};

} /* namespace NN */
#endif /* GENERATOR_H_ */
