/*
 * Generator.cpp
 *
 *  Created on: Jul 2, 2014
 *      Author: shaun
 */

#include "Generator.h"

namespace NN
{

Generator::Generator(size_t population_size, vector<float> std_devs)
: m_pop_size(population_size), m_std_devs(std_devs)
{
}

Generator::~Generator()
{
}

vector<shared_ptr<NeuralNet>>
Generator::generate(vector<shared_ptr<NeuralNet>> gene_pool)
{
    size_t base_size = gene_pool.size();
    size_t population = base_size; // Maintain copies of originals
    while(population < m_pop_size)
    {
        int something = 0;
        gene_pool.push_back(gene_pool[something % base_size]);

        ++population;

        // For each std_dev, make a copy of each neural net
        // When end of std_devs has been reached, start over
        // do until population = pop_size
    }

    return gene_pool;
}

} /* namespace NN */
