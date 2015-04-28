/*
 * main.cpp
 *
 *  Created on: Jun 2, 2014
 *      Author: shaun
 */

// TODO: profile libneuralnet to find bottlenecks
#define CATCH_CONFIG_MAIN

#include "catch.hpp"
#include "NeuralNet.h"
using NN::NeuralNet;
using NN::CheckersNet;

#include "Synapse.h"
using NN::SigmoidSynapse;

#include "MoveFinder.h"
using NN::MoveFinder;
using NN::CheckersMoveFinder;

#include "ABSearch.h"
using NN::AB_Search;

#include <limits>
#include <memory>
using std::shared_ptr;
using std::make_shared;

#include <random>
using std::srand;
using std::rand;

#include <chrono>
using std::chrono::high_resolution_clock;
using std::chrono::duration;
using std::chrono::duration_cast;

#include <vector>
using std::vector;

#include <set>
using std::set;

namespace
{
const float MAX_FLOAT =  std::numeric_limits<float>::max();
const float MIN_FLOAT = -std::numeric_limits<float>::max();
}


SCENARIO("Synapses", "synapses")
{
    GIVEN("A SigmoidSynapse initialized with coeff = 1.0")
    {
        SigmoidSynapse synapse(1);
        float value;

        WHEN("The synapse is queried with a value of 0")
        {
            value = synapse(0);

            THEN("The synapse should return 0.5")
            {
                REQUIRE(value < 0.5000001);
                REQUIRE(value > 0.4999999);
            }
        }

        WHEN("The synapse is queried with a value of 1")
        {
            value = synapse(1);

            THEN("The synapse should return 2/3")
            {
                REQUIRE(value < 0.6666667);
                REQUIRE(value > 0.6666666);
            }
        }

        WHEN("The synapse is queried with a value of -1")
        {
            value = synapse(-1);

            THEN("The synapse should return 1/3")
            {
                REQUIRE(value < 0.3333334);
                REQUIRE(value > 0.3333333);
            }
        }

        WHEN("The synapse is queried with a value of 2")
        {
            value = synapse(2);

            THEN("The synapse should return 0.75")
            {
                REQUIRE(value < 0.7500001);
                REQUIRE(value > 0.7499999);
            }
        }

        WHEN("The synapse is queried with a value of -2")
        {
            value = synapse(-2);

            THEN("The synapse should return 0.2")
            {
                REQUIRE(value < 0.2500001);
                REQUIRE(value > 0.2499999);
            }
        }

        WHEN("The synapse is queried with a value greater than 9")
        {
            THEN("The synapse should return a value greater than 0.9")
            {
                REQUIRE(synapse(9) > 0.9);
                REQUIRE(synapse(90) > 0.9);
                REQUIRE(synapse(900) > 0.9);
            }
        }

        WHEN("The synapse is queried with a value less than -9")
        {
            THEN("The synapse should return a value less than 0.1")
            {
                REQUIRE(synapse(-9) < 0.1);
                REQUIRE(synapse(-90) < 0.1);
                REQUIRE(synapse(-900) < 0.1);
            }
        }

        WHEN("The synapse is queried with the maximum value")
        {
            THEN("The synapse should return a value less than 1.0")
            {
                REQUIRE(synapse(MAX_FLOAT) <= 1);
            }
        }

        WHEN("The synapse is queried with the minimum value")
        {
            THEN("The synapse should return a value greater than 0.0")
            {
                REQUIRE(synapse(MIN_FLOAT) >= 0);
            }
        }
    } // End SigmoidSynapse coeff = 1

    GIVEN("A SigmoidSynapse initialized with coeff = 16.0")
    {
        SigmoidSynapse synapse(16);
        float value;

        WHEN("The synapse is queried with a value of 0")
        {
            value = synapse(0);

            THEN("The synapse should return 0.5")
            {
                REQUIRE(value < 0.5000001);
                REQUIRE(value > 0.4999999);
            }
        }

        WHEN("The synapse is queried with a value of 1")
        {
            value = synapse(1);

            THEN("The synapse should return 0.9444...")
            {
                REQUIRE(value < 0.9444445);
                REQUIRE(value > 0.9444443);
            }
        }

        WHEN("The synapse is queried with a value of -1")
        {
            value = synapse(-1);

            THEN("The synapse should return 0.0555...")
            {
                REQUIRE(value < 0.0555556);
                REQUIRE(value > 0.0555554);
            }
        }

        WHEN("The synapse is queried with a value of 2")
        {
            value = synapse(2);

            THEN("The synapse should return ~ 0.970588")
            {
                REQUIRE(value < 0.970589);
                REQUIRE(value > 0.970588);
            }
        }

        WHEN("The synapse is queried with a value of -2")
        {
            value = synapse(-2);

            THEN("The synapse should return ~ 0.029412")
            {
                REQUIRE(value < 0.029412);
                REQUIRE(value > 0.029411);
            }
        }

        WHEN("The synapse is queried with a value greater than 9")
        {
            THEN("The synapse should return a value greater than 0.99")
            {
                REQUIRE(synapse(9) > 0.99);
                REQUIRE(synapse(90) > 0.99);
                REQUIRE(synapse(900) > 0.99);
            }
        }

        WHEN("The synapse is queried with a value less than -9")
        {
            THEN("The synapse should return a value less than 0.01")
            {
                REQUIRE(synapse(-9) < 0.01);
                REQUIRE(synapse(-90) < 0.01);
                REQUIRE(synapse(-900) < 0.01);
            }
        }

        WHEN("The synapse is queried with the maximum value")
        {
            THEN("The synapse should return a value approximately equal to 1.0")
            {
                REQUIRE(synapse(MAX_FLOAT) <= 1);
                REQUIRE(synapse(MAX_FLOAT) >= (1 - 2e-40));
            }
        }

        WHEN("The synapse is queried with the minimum value")
        {
            THEN("The synapse should return a value approximately equal to 0.0")
            {
                REQUIRE(synapse(MIN_FLOAT) >= 0);
                REQUIRE(synapse(MIN_FLOAT) <= 2e-40);
            }
        }
    } // End SigmoidSynapse coeff = 16
} // End "Synapses"



SCENARIO("Constructing a neural net", "constructors")
{
	GIVEN("A vector")
	{
	    vector<size_t> node_structure1 = {4, 8, 4, 1};
        vector<size_t> node_structure2 = {30, 128, 66, 25, 48, 1};

		WHEN("A NeuralNet is created using the vector")
	    {
            NeuralNet test1(node_structure1, make_shared<SigmoidSynapse>(1));
            NeuralNet test2(node_structure2, make_shared<SigmoidSynapse>(1));

            THEN("The NeuralNet's node structure should match the vector")
		    {
			    REQUIRE(test1.structure() == node_structure1);
			    REQUIRE(test2.structure() == node_structure2);
			}
		}

        WHEN("A CheckersNet is created using the vector")
        {
            CheckersNet test1(node_structure1, make_shared<SigmoidSynapse>(1));
            CheckersNet test2(node_structure2, make_shared<SigmoidSynapse>(1));

            THEN("The CheckersNet's node structure should match the vector")
            {
                REQUIRE(test1.structure() == node_structure1);
                REQUIRE(test2.structure() == node_structure2);
            }
        }

		GIVEN("A set of random inputs")
		{
		    srand(time(0));
		    vector<vector<float> > inputs1;
            vector<vector<float> > inputs2;

            for(auto ii = 0u; ii < 10; ++ii)
		    {
		        vector<float> input;
	            for(auto ii = 0u; ii < 4; ++ii)
	            {
	                input.push_back( (rand()%1000)/1000.0 );
	            }
	            inputs1.push_back(input);

	            input.clear();
	            for(auto ii = 0u; ii < 30; ++ii)
                {
                    input.push_back( (rand()%1000)/1000.0 );
                }
                inputs2.push_back(input);
		    }


	        WHEN("Multiple NeuralNets are created with the same vector")
	        {
	            NeuralNet test1(node_structure1, make_shared<SigmoidSynapse>(1));
	            NeuralNet test2(node_structure1, make_shared<SigmoidSynapse>(1));

	            NeuralNet test3(node_structure2, make_shared<SigmoidSynapse>(1));
	            NeuralNet test4(node_structure2, make_shared<SigmoidSynapse>(1));

	            THEN("They should produce different results from the same input")
	            {
	                bool different1 = true;
	                for(auto input : inputs1)
	                {
	                    different1 = ( test1.evaluate(input) != test2.evaluate(input) )
	                                 && different1;
	                }

	                bool different2 = true;
                    for(auto input : inputs2)
                    {
                        different2 = ( test3.evaluate(input) != test4.evaluate(input) )
                                     && different2;
                    }

	                REQUIRE(different1);
	                REQUIRE(different2);
	            }
	        }

	        WHEN("A NeuralNet is created from an existing neural network")
	        {
                NeuralNet test1(node_structure1, make_shared<SigmoidSynapse>(1));
                NeuralNet test2(test1);

                NeuralNet test3(node_structure2, make_shared<SigmoidSynapse>(1));
                NeuralNet test4(test3);

                THEN("They should produce the same results from the same input")
                {
                    bool different1 = false;
                    for(auto input : inputs1)
                    {
                        different1 = ( test1.evaluate(input) != test2.evaluate(input) )
                                     || different1;
                    }

                    bool different2 = false;
                    for(auto input : inputs2)
                    {
                        different2 = ( test3.evaluate(input) != test4.evaluate(input) )
                                     || different2;
                    }

                    REQUIRE(!different1);
                    REQUIRE(!different2);
                }
	        }

            WHEN("Multiple CheckersNets are created with the same vector")
            {
                CheckersNet test1(node_structure1, make_shared<SigmoidSynapse>(1));
                CheckersNet test2(node_structure1, make_shared<SigmoidSynapse>(1));

                CheckersNet test3(node_structure2, make_shared<SigmoidSynapse>(1));
                CheckersNet test4(node_structure2, make_shared<SigmoidSynapse>(1));

                THEN("They should produce different results from the same input")
                {
                    bool different1 = true;
                    for(auto input : inputs1)
                    {
                        different1 = ( test1.evaluate(input) != test2.evaluate(input) )
                                     && different1;
                    }

                    bool different2 = true;
                    for(auto input : inputs2)
                    {
                        different2 = ( test3.evaluate(input) != test4.evaluate(input) )
                                     && different2;
                    }

                    REQUIRE(different1);
                    REQUIRE(different2);
                }
            }

            WHEN("A CheckersNet is created from an existing neural network")
            {
                CheckersNet test1(node_structure1, make_shared<SigmoidSynapse>(1));
                CheckersNet test2(test1);

                CheckersNet test3(node_structure2, make_shared<SigmoidSynapse>(1));
                CheckersNet test4(test3);

                THEN("They should produce the same results from the same input")
                {
                    bool different1 = false;
                    for(auto input : inputs1)
                    {
                        different1 = ( test1.evaluate(input) != test2.evaluate(input) )
                                     || different1;
                    }

                    bool different2 = false;
                    for(auto input : inputs2)
                    {
                        different2 = ( test3.evaluate(input) != test4.evaluate(input) )
                                     || different2;
                    }

                    REQUIRE(!different1);
                    REQUIRE(!different2);
                }
            }
		} // End "Given a set of random inputs"
	} // End "Given a vector"
} // End "Constructing a neural net"



SCENARIO("Evaluating a set of inputs with a neural net", "evaluate")
{
    GIVEN("A set of random inputs")
    {
        srand(time(0));
        vector<vector<float> > inputs;

        for(auto ii = 0u; ii < 10; ++ii)
        {
            vector<float> input;
            for(auto ii = 0u; ii < 4; ++ii)
            {
                input.push_back( (rand()%1000)/1000.0 );
            }

            inputs.push_back(input);
        }

        vector<size_t> node_structure = {30, 128, 66, 25, 48, 1};

        GIVEN("A NeuralNet")
        {
            NeuralNet test(node_structure, make_shared<SigmoidSynapse>(1));

            WHEN("Provided with different inputs")
            {
                THEN("evaluate() should produce different results")
                {
                    for(auto ii = 0u; ii < inputs.size()-1; ++ii)
                    {
                        REQUIRE(test.evaluate(inputs[ii]) != test.evaluate(inputs[ii+1]));
                    }
                }
            } // When provided with different inputs

            WHEN("Provided with the same input")
            {
                THEN("evaluate() should produce the same result")
                {
                    for(auto ii = 0u; ii < inputs.size(); ++ii)
                    {
                        REQUIRE(test.evaluate(inputs[ii]) == test.evaluate(inputs[ii]));
                    }
                }
            } // When provided with the same input
        } // Given a NeuralNet

        GIVEN("A CheckersNet")
        {
            CheckersNet test(node_structure, make_shared<SigmoidSynapse>(1));

            WHEN("Provided with different inputs")
            {
                THEN("evaluate() should produce different results")
                {
                    test.player(CheckersNet::BLACK);
                    for(auto ii = 0u; ii < inputs.size()-1; ++ii)
                    {
                        REQUIRE(test.evaluate(inputs[ii]) != test.evaluate(inputs[ii+1]));
                    }

                    test.player(CheckersNet::RED);
                    for(auto ii = 0u; ii < inputs.size()-1; ++ii)
                    {
                        REQUIRE(test.evaluate(inputs[ii]) != test.evaluate(inputs[ii+1]));
                    }
                }
            } // When provided with different inputs

            WHEN("Provided with the same input")
            {
                THEN("evaluate() should produce the same result")
                {
                    test.player(CheckersNet::BLACK);
                    for(auto ii = 0u; ii < inputs.size(); ++ii)
                    {
                        REQUIRE(test.evaluate(inputs[ii]) == test.evaluate(inputs[ii]));
                    }

                    test.player(CheckersNet::RED);
                    for(auto ii = 0u; ii < inputs.size(); ++ii)
                    {
                        REQUIRE(test.evaluate(inputs[ii]) == test.evaluate(inputs[ii]));
                    }
                }
            } // When provided with the same input

            WHEN("The same input is evaluated as different players")
            {
                CheckersNet black(test);
                black.player(CheckersNet::BLACK);

                CheckersNet red(test);
                red.player(CheckersNet::RED);

                THEN("The results should be mirrored about a median")
                {
                    SigmoidSynapse s(1);
                    float medianx2 = s(MAX_FLOAT) + s(MIN_FLOAT);

                    for(auto ii = 0u; ii < inputs.size(); ++ii)
                    {
                        REQUIRE((black.evaluate(inputs[ii]) + red.evaluate(inputs[ii])) == medianx2);
                    }

                    //TODO: test CheckersNet::evaluate with different synapses
                }
            } // When the same input is evaluated as different players
        } // Given a CheckersNet
    } // Given a set of random inputs
}



SCENARIO("CheckersMoveFinder", "moveFinder")
{
    const bool BLACK = true;
    const bool RED = false;

    typedef vector<float> Board;

    GIVEN("A CheckersMoveFinder")
    {
        CheckersMoveFinder finder;

        WHEN("The board contains only friendly pawns")
        {
            Board board1 =
            {1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

            Board board2 =
            {0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

            Board board3 =
            {0,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

            Board board4 =
            {0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

            Board board5 =
            {0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

            Board board6 =
            {0,0,0,0,0,1,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

            Board board7 =
            {1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

            Board board8 =
            {0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

            Board board9 =
            {0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

            Board board10 =
            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,1,1,1};

            Board board11 =
            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,-1};

            Board board12 =
            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,-1,0};

            Board board13 =
            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,-1,-1,0};

            Board board14 =
            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,-1,0,0,0,0};

            Board board15 =
            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,-1,0,0,0,0,0};

            Board board16 =
            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,-1,-1,0,0,0,0,0};

            Board board17 =
            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1};

            Board board18 =
            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,-1,-1,-1,-1,-1,-1,-1,0,0,0,0};

            Board board19 =
            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,-1,-1,-1,-1,-1,-1,-1,0,0,0,0,0,0,0,0};

            Board board20 =
            {-1,-1,-1,-1,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

            THEN("It should produce a list containing all possible moves")
            {
                set<Board> possibleMoves =
                {{0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}};

                set<Board> moves;
                for(auto vec : finder.possibleMoves(board1, BLACK))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                possibleMoves =
                {{0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}};

                moves.clear();
                for(auto vec : finder.possibleMoves(board2, BLACK))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                possibleMoves =
                {{0,0,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,1,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,1,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,1,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}};

                moves.clear();
                for(auto vec : finder.possibleMoves(board3, BLACK))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                possibleMoves =
                {{0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}};

                moves.clear();
                for(auto vec : finder.possibleMoves(board4, BLACK))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                possibleMoves =
                {{0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}};

                moves.clear();
                for(auto vec : finder.possibleMoves(board5, BLACK))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                possibleMoves =
                {{0,0,0,0,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,0,0,0,0,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,0,0,0,0,1,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}};

                moves.clear();
                for(auto vec : finder.possibleMoves(board6, BLACK))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                possibleMoves =
                {{1,1,1,1,1,1,1,1,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {1,1,1,1,1,1,1,1,0,1,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {1,1,1,1,1,1,1,1,1,0,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {1,1,1,1,1,1,1,1,1,0,1,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {1,1,1,1,1,1,1,1,1,1,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {1,1,1,1,1,1,1,1,1,1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}};

                moves.clear();
                for(auto vec : finder.possibleMoves(board7, BLACK))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                possibleMoves =
                {{0,0,0,0,1,1,1,1,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,0,0,0,1,1,1,1,0,1,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,0,0,0,1,1,1,1,1,0,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,0,0,0,1,1,1,1,1,0,1,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,0,0,0,1,1,1,1,1,1,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,0,0,0,1,1,1,1,1,1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}};

                moves.clear();
                for(auto vec : finder.possibleMoves(board8, BLACK))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                possibleMoves =
                {{0,0,0,0,0,0,0,0,1,1,1,1,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,0,0,0,0,0,0,0,1,1,1,1,1,0,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,0,0,0,0,0,0,0,1,1,1,1,1,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0}};

                moves.clear();
                for(auto vec : finder.possibleMoves(board9, BLACK))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                possibleMoves =
                {{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,1,1},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1,1,1}};

                moves.clear();
                for(auto vec : finder.possibleMoves(board10, BLACK))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                possibleMoves =
                {{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,-1},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,-1,0,0,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,-1,0,0,0}};

                moves.clear();
                for(auto vec : finder.possibleMoves(board11, RED))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                possibleMoves =
                {{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,-1,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,-1,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,-1,0,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,-1,0,0}};

                moves.clear();
                for(auto vec : finder.possibleMoves(board12, RED))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                possibleMoves =
                {{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,-1,-1,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,-1,-1,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,-1,0,0,0,0,-1,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,-1,0,0,-1,0,0}};

                moves.clear();
                for(auto vec : finder.possibleMoves(board13, RED))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                possibleMoves =
                {{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,-1,0,0,0,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,-1,0,0,0,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,-1,0,0,0,0,0,0,0}};

                moves.clear();
                for(auto vec : finder.possibleMoves(board14, RED))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                possibleMoves =
                {{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,-1,0,0,0,0,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,-1,0,0,0,0,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,-1,0,0,0,0,0,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,-1,0,0,0,0,0,0}};

                moves.clear();
                for(auto vec : finder.possibleMoves(board15, RED))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                possibleMoves =
                {{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,-1,-1,0,0,0,0,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,-1,-1,0,0,0,0,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,-1,0,0,0,-1,0,0,0,0,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,-1,0,-1,0,0,0,0,0,0}};

                moves.clear();
                for(auto vec : finder.possibleMoves(board16, RED))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                possibleMoves =
                {{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,-1,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,-1,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,-1,-1,0,-1,-1,-1,-1,-1,-1,-1,-1,-1},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,-1,-1,0,-1,-1,-1,-1,-1,-1,-1,-1,-1},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,-1,-1,-1,0,-1,-1,-1,-1,-1,-1,-1,-1},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,-1,-1,-1,0,-1,-1,-1,-1,-1,-1,-1,-1}};

                moves.clear();
                for(auto vec : finder.possibleMoves(board17, RED))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                possibleMoves =
                {{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,-1,-1,-1,-1,-1,-1,-1,0,0,0,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,-1,0,-1,-1,-1,-1,-1,-1,0,0,0,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,-1,0,-1,-1,-1,-1,-1,-1,0,0,0,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,-1,-1,0,-1,-1,-1,-1,-1,0,0,0,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,-1,-1,0,-1,-1,-1,-1,-1,0,0,0,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,-1,-1,-1,0,-1,-1,-1,-1,0,0,0,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,-1,-1,-1,0,-1,-1,-1,-1,0,0,0,0}};

                moves.clear();
                for(auto vec : finder.possibleMoves(board18, RED))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                possibleMoves =
                {{0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,-1,-1,-1,-1,-1,-1,-1,0,0,0,0,0,0,0,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,-1,-1,-1,-1,-1,-1,-1,0,0,0,0,0,0,0,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,-1,0,-1,-1,-1,-1,-1,-1,0,0,0,0,0,0,0,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,-1,0,-1,-1,-1,-1,-1,-1,0,0,0,0,0,0,0,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,-1,-1,0,-1,-1,-1,-1,-1,0,0,0,0,0,0,0,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,-1,-1,0,-1,-1,-1,-1,-1,0,0,0,0,0,0,0,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,-1,-1,-1,0,-1,-1,-1,-1,0,0,0,0,0,0,0,0}};

                moves.clear();
                for(auto vec : finder.possibleMoves(board19, RED))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                possibleMoves =
                {{-1,-1,-1,-1,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {-1,-1,-1,-1,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}};

                moves.clear();
                for(auto vec : finder.possibleMoves(board20, RED))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);
            }
        } // Only friendly pawns

        WHEN("A pawn moves into the last row")
        {
            Board kingmakerBlack =
            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0};

            Board kingmakerRed =
            {0,0,0,0,-1,-1,-1,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

            THEN("It should become a king")
            {
                set<Board> possibleMoves =
                {{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1.5,0,0,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1.5,0,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,0,1.5,0,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,0,0,1.5,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,0,0,1.5,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,0,0,0,1.5},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,1.5}};

                set<Board> moves;
                for(auto vec : finder.possibleMoves(kingmakerBlack, BLACK))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                possibleMoves =
                {{0,0,0,-1.5,-1,-1,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,0,-1.5,0,-1,-1,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,0,-1.5,0,-1,-1,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,-1.5,0,0,-1,-1,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,-1.5,0,0,-1,0,-1,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {-1.5,0,0,0,-1,0,-1,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {-1.5,0,0,0,0,-1,-1,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}};

                moves.clear();
                for(auto vec : finder.possibleMoves(kingmakerRed, RED))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);
            }
        } // Pawns moving into last row

        WHEN("There are enemy pawns on the board")
        {
            Board board1 =
            {0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,-1,-1,-1,0,0,0,0,0,0,0,0,0,0,0,0,0};

            Board board2 =
            {0,0,0,0,0,0,0,0,-1,-1,0,0,0,1,0,0,-1,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

            Board board3 =
            {0,0,0,0,0,0,0,0,-1,-1,0,0,0,1,1,1,-1,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

            Board board4 =
            {0,0,0,0,0,0,0,0,0,0,0,0,-1,-1,0,0,1,1,1,0,-1,-1,0,0,0,0,0,0,0,0,0,0};

            Board board5 =
            {0,0,0,0,0,0,0,0,0,0,-1,-1,0,1,1,1,0,0,-1,-1,0,0,0,0,0,0,0,0,0,0,0,0};

            Board board6 =
            {-1,-1,1,0,0,1,-1,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

            Board board7 =
            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,-1,-1,1};

            Board board8 =
            {0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0};

            Board board9 =
            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,-1,0,0,0,1,1,0,0,0,0,0,0,0,0};

            Board board10 =
            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,-1,-1,-1,0,0,0,1,1,0,0,0,0,0,0,0,0};

            Board board11 =
            {0,0,0,0,0,0,0,0,0,0,1,1,0,-1,-1,-1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0};

            Board board12 =
            {0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,-1,-1,-1,0,1,1,0,0,0,0,0,0,0,0,0,0};

            Board board13 =
            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,-1,0,0,-1,1,1};

            Board board14 =
            {-1,1,1,0,0,0,-1,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

            THEN("They should consider only jumps when available")
            {
                set<Board> possibleMoves =
                {{0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,-1,-1,-1,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,-1,-1,-1,0,0,0,0,0,0,0,0,0,0,0,0,0}};

                set<Board> moves;
                for(auto vec : finder.possibleMoves(board1, BLACK))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                possibleMoves =
                {{0,0,0,0,0,0,0,0,-1,-1,0,0,0,0,0,0,0,-1,0,0,1,0,0,0,0,0,0,0,0,0,0,0},
                 {0,0,0,0,0,0,0,0,-1,-1,0,0,0,0,0,0,-1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0}};

                moves.clear();
                for(auto vec : finder.possibleMoves(board2, BLACK))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                possibleMoves =
                {{0,0,0,0,0,0,0,0,-1,-1,0,0,0,0,1,1,0,-1,0,0,1,0,0,0,0,0,0,0,0,0,0,0},
                 {0,0,0,0,0,0,0,0,-1,-1,0,0,0,0,1,1,-1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0},
                 {0,0,0,0,0,0,0,0,-1,-1,0,0,0,1,0,1,-1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0}};

                moves.clear();
                for(auto vec : finder.possibleMoves(board3, BLACK))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                possibleMoves =
                {{0,0,0,0,0,0,0,0,0,0,0,0,-1,-1,0,0,0,1,1,0,-1,0,0,0,0,1,0,0,0,0,0,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,-1,-1,0,0,1,0,1,0,-1,0,0,0,1,0,0,0,0,0,0,0}};

                moves.clear();
                for(auto vec : finder.possibleMoves(board4, BLACK))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                possibleMoves =
                {{0,0,0,0,0,0,0,0,0,0,-1,-1,0,1,0,1,0,0,0,-1,0,0,0,1,0,0,0,0,0,0,0,0},
                 {0,0,0,0,0,0,0,0,0,0,-1,-1,0,1,1,0,0,0,0,-1,0,0,1,0,0,0,0,0,0,0,0,0}};

                moves.clear();
                for(auto vec : finder.possibleMoves(board5, BLACK))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                possibleMoves =
                {{-1,-1,0,0,0,1,0,0,-1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {-1,-1,1,0,0,0,-1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}};

                moves.clear();
                for(auto vec : finder.possibleMoves(board6, BLACK))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                possibleMoves =
                {{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1.5,-1,-1,1}};

                moves.clear();
                for(auto vec : finder.possibleMoves(board7, BLACK))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                possibleMoves =
                {{0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0}};

                moves.clear();
                for(auto vec : finder.possibleMoves(board8, RED))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                possibleMoves =
                {{0,0,0,0,0,0,0,0,0,0,0,-1,0,0,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0},
                 {0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0}};

                moves.clear();
                for(auto vec : finder.possibleMoves(board9, RED))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                possibleMoves =
                {{0,0,0,0,0,0,0,0,0,0,0,-1,0,0,1,0,-1,-1,0,0,0,0,1,1,0,0,0,0,0,0,0,0},
                 {0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,1,-1,-1,0,0,0,0,1,1,0,0,0,0,0,0,0,0},
                 {0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,1,-1,0,-1,0,0,0,1,1,0,0,0,0,0,0,0,0}};

                moves.clear();
                for(auto vec : finder.possibleMoves(board10, RED))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                possibleMoves =
                {{0,0,0,0,0,0,-1,0,0,0,0,1,0,-1,-1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,0,0,0,0,0,0,-1,0,0,0,1,0,-1,0,-1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0}};

                moves.clear();
                for(auto vec : finder.possibleMoves(board11, RED))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                possibleMoves =
                {{0,0,0,0,0,0,0,0,-1,0,0,0,1,0,0,0,-1,0,-1,0,1,1,0,0,0,0,0,0,0,0,0,0},
                 {0,0,0,0,0,0,0,0,0,-1,0,0,1,0,0,0,0,-1,-1,0,1,1,0,0,0,0,0,0,0,0,0,0}};

                moves.clear();
                for(auto vec : finder.possibleMoves(board12, RED))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                possibleMoves =
                {{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,1,0,0,-1,0,0,0,1,1},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,1,0,0,0,-1,1,1}};

                moves.clear();
                for(auto vec : finder.possibleMoves(board13, RED))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                possibleMoves =
                {{-1,1,1,-1.5,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}};

                moves.clear();
                for(auto vec : finder.possibleMoves(board14, RED))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);
            }
        } // When there are enemy pawns on the board

        WHEN("All pawns are blocked")
        {
            Board board1 =
            {0,0,0,0,1,0,0,0,-1,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

            Board board2 =
            {0,0,0,0,0,0,0,0,1,0,0,0,-1,-1,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

            Board board3 =
            {0,0,0,0,0,0,0,1,0,0,-1,-1,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

            Board board4 =
            {0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,-1,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0};

            Board board5 =
            {0,0,0,0,0,0,0,0,0,1,0,0,0,-1,-1,0,-1,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0};

            Board board6 =
            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1};

            Board board7 =
            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,-1,0,0,0,0};

            Board board8 =
            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1,0,0,0,-1,0,0,0,0,0,0,0,0};

            Board board9 =
            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1,0,0,-1,0,0,0,0,0,0,0};

            Board board10 =
            {0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0};

            Board board11 =
            {0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,1,0,0,0,-1,0,0,0,0,0,0,0,0,0};

            Board board12 =
            {-1,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

            THEN("They should not move")
            {
                set<Board> possibleMoves;

                set<Board> moves;
                for(auto vec : finder.possibleMoves(board1, BLACK))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                moves.clear();
                for(auto vec : finder.possibleMoves(board2, BLACK))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                moves.clear();
                for(auto vec : finder.possibleMoves(board3, BLACK))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                moves.clear();
                for(auto vec : finder.possibleMoves(board4, BLACK))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                moves.clear();
                for(auto vec : finder.possibleMoves(board5, BLACK))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                moves.clear();
                for(auto vec : finder.possibleMoves(board6, BLACK))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                moves.clear();
                for(auto vec : finder.possibleMoves(board7, RED))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                moves.clear();
                for(auto vec : finder.possibleMoves(board8, RED))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                moves.clear();
                for(auto vec : finder.possibleMoves(board9, RED))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                moves.clear();
                for(auto vec : finder.possibleMoves(board10, RED))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                moves.clear();
                for(auto vec : finder.possibleMoves(board11, RED))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                moves.clear();
                for(auto vec : finder.possibleMoves(board12, RED))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);
            } // Then they should not move
        } // When all pawns are blocked

        WHEN("A pawn jumps to the last row")
        {
            Board kingmakerBlack =
            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,-1,-1,-1,-1,0,0,0,0};

            Board kingmakerRed =
            {0,0,0,0,1,1,1,1,-1,-1,-1,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

            THEN("It should become a king")
            {
                set<Board> possibleMoves =
                {{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,-1,-1,-1,0,1.5,0,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,0,-1,-1,-1,1.5,0,0,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,-1,0,-1,-1,0,0,1.5,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,-1,0,-1,-1,0,1.5,0,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,-1,-1,0,-1,0,0,0,1.5},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,-1,-1,0,-1,0,0,1.5,0}};

                set<Board> moves;
                for(auto vec : finder.possibleMoves(kingmakerBlack, BLACK))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                possibleMoves =
                {{0,0,-1.5,0,1,1,1,0,-1,-1,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,0,0,-1.5,1,1,1,0,-1,-1,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,-1.5,0,0,1,1,0,1,-1,-1,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,0,-1.5,0,1,1,0,1,-1,0,-1,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {-1.5,0,0,0,1,0,1,1,-1,0,-1,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,-1.5,0,0,1,0,1,1,0,-1,-1,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}};

                moves.clear();
                for(auto vec : finder.possibleMoves(kingmakerRed, RED))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);
            }
        } // When a pawn jumps to the last row

        WHEN("The board contains kings")
        {
            Board board1 =
            {1.5,0,0,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

            Board board2 =
            {0,1.5,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

            Board board3 =
            {0,1.5,1.5,0,0,0,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

            Board board4 =
            {0,0,0,0,1.5,0,0,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

            Board board5 =
            {0,0,0,0,0,1.5,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

            Board board6 =
            {0,0,0,0,0,1.5,1.5,0,0,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

            Board board7 =
            {1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

            Board board8 =
            {0,0,0,0,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

            Board board9 =
            {0,0,0,0,0,0,0,0,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

            Board board10 =
            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.5,0,0,0,0,0,1.5,1.5,1.5,1.5};

            Board board11 =
            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.5,0,0,-1.5};

            Board board12 =
            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.5,-1.5,0};

            Board board13 =
            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.5,0,0,0,-1.5,-1.5,0};

            Board board14 =
            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.5,0,0,-1.5,0,0,0,0};

            Board board15 =
            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.5,-1.5,0,0,0,0,0};

            Board board16 =
            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.5,0,0,-1.5,-1.5,0,0,0,0,0};

            Board board17 =
            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.5,-1.5,-1.5,-1.5,-1.5,-1.5,-1.5,-1.5,-1.5,-1.5,-1.5,-1.5};

            Board board18 =
            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.5,-1.5,-1.5,-1.5,-1.5,-1.5,-1.5,-1.5,0,0,0,0};

            Board board19 =
            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.5,-1.5,-1.5,-1.5,-1.5,-1.5,-1.5,-1.5,0,0,0,0,0,0,0,0};

            Board board20 =
            {-1.5,-1.5,-1.5,-1.5,0,0,0,0,0,-1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

            THEN("They should be able to move backwards")
            {
                set<Board> possibleMoves =
                {{0,0,0,1.5,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,0,0,1.5,0,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {1.5,0,0,0,0,0,0,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}};

                set<Board> moves;
                for(auto vec : finder.possibleMoves(board1, BLACK))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                possibleMoves =
                {{0,0,1.5,0,0,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,0,1.5,0,0,0,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,1.5,0,0,0,0,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,1.5,0,0,0,0,0,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}};

                moves.clear();
                for(auto vec : finder.possibleMoves(board2, BLACK))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                possibleMoves =
                {{0,0,1.5,0,0,1.5,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,1.5,0,0,0,0,1.5,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,1.5,1.5,0,0,0,0,0,0,0,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,1.5,1.5,0,0,0,0,0,0,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}};

                moves.clear();
                for(auto vec : finder.possibleMoves(board3, BLACK))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                possibleMoves =
                {{0,0,0,0,0,0,0,1.5,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {1.5,0,0,0,0,0,0,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,0,0,0,1.5,0,0,0,0,0,0,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,0,0,0,1.5,0,0,0,0,0,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,0,0,1.5,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,0,1.5,0,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}};

                moves.clear();
                for(auto vec : finder.possibleMoves(board4, BLACK))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                possibleMoves =
                {{0,0,0,0,0,0,1.5,0,0,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,0,0,0,0,0,1.5,0,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,1.5,0,0,0,0,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {1.5,0,0,0,0,0,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,0,0,0,0,1.5,0,0,0,0,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,0,0,0,0,1.5,0,0,0,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,0,1.5,0,0,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,1.5,0,0,0,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}};

                moves.clear();
                for(auto vec : finder.possibleMoves(board5, BLACK))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                possibleMoves =
                {{0,0,0,0,0,0,1.5,0,1.5,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,1.5,0,0,0,0,1.5,0,0,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {1.5,0,0,0,0,0,1.5,0,0,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,0,0,0,0,1.5,0,0,0,1.5,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,0,1.5,0,0,1.5,0,0,0,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,1.5,0,0,0,1.5,0,0,0,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,0,0,0,0,1.5,1.5,0,0,0,0,0,0,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,0,0,0,0,1.5,1.5,0,0,0,0,0,0,0,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}};

                moves.clear();
                for(auto vec : finder.possibleMoves(board6, BLACK))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                possibleMoves =
                {{1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,0,1.5,1.5,1.5,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,0,1.5,1.5,1.5,0,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,0,1.5,1.5,0,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,0,1.5,1.5,0,0,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,0,1.5,0,0,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,0,1.5,0,0,0,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,0,0,0,0,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}};

                moves.clear();
                for(auto vec : finder.possibleMoves(board7, BLACK))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                possibleMoves =
                {{1.5,0,0,0,0,1.5,1.5,1.5,1.5,1.5,1.5,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,1.5,0,0,1.5,0,1.5,1.5,1.5,1.5,1.5,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {1.5,0,0,0,1.5,0,1.5,1.5,1.5,1.5,1.5,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,0,1.5,0,1.5,1.5,0,1.5,1.5,1.5,1.5,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,1.5,0,0,1.5,1.5,0,1.5,1.5,1.5,1.5,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,0,0,1.5,1.5,1.5,1.5,0,1.5,1.5,1.5,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,0,1.5,0,1.5,1.5,1.5,0,1.5,1.5,1.5,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,0,0,0,1.5,1.5,1.5,1.5,0,1.5,1.5,1.5,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,0,0,0,1.5,1.5,1.5,1.5,0,1.5,1.5,1.5,0,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,0,0,0,1.5,1.5,1.5,1.5,1.5,0,1.5,1.5,0,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,0,0,0,1.5,1.5,1.5,1.5,1.5,0,1.5,1.5,0,0,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,0,0,0,1.5,1.5,1.5,1.5,1.5,1.5,0,1.5,0,0,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,0,0,0,1.5,1.5,1.5,1.5,1.5,1.5,0,1.5,0,0,0,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,0,0,0,1.5,1.5,1.5,1.5,1.5,1.5,1.5,0,0,0,0,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}};

                moves.clear();
                for(auto vec : finder.possibleMoves(board8, BLACK))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                possibleMoves =
                {{0,0,0,0,1.5,0,0,0,0,1.5,1.5,1.5,1.5,1.5,1.5,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,0,0,0,0,1.5,0,0,0,1.5,1.5,1.5,1.5,1.5,1.5,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,0,0,0,0,1.5,0,0,1.5,0,1.5,1.5,1.5,1.5,1.5,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,0,0,0,0,0,1.5,0,1.5,0,1.5,1.5,1.5,1.5,1.5,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,0,0,0,0,0,1.5,0,1.5,1.5,0,1.5,1.5,1.5,1.5,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,0,0,0,0,0,0,1.5,1.5,1.5,0,1.5,1.5,1.5,1.5,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,0,0,0,0,0,0,1.5,1.5,1.5,1.5,0,1.5,1.5,1.5,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,0,0,0,0,0,0,0,1.5,1.5,1.5,1.5,0,1.5,1.5,1.5,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,0,0,0,0,0,0,0,1.5,1.5,1.5,1.5,1.5,0,1.5,1.5,0,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,0,0,0,0,0,0,0,1.5,1.5,1.5,1.5,1.5,0,1.5,1.5,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,0,0,0,0,0,0,0,1.5,1.5,1.5,1.5,1.5,1.5,0,1.5,0,0,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,0,0,0,0,0,0,0,1.5,1.5,1.5,1.5,1.5,1.5,0,1.5,0,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,0,0,0,0,0,0,0,1.5,1.5,1.5,1.5,1.5,1.5,1.5,0,0,0,0,1.5,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,0,0,0,0,0,0,0,1.5,1.5,1.5,1.5,1.5,1.5,1.5,0,0,0,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0}};

                moves.clear();
                for(auto vec : finder.possibleMoves(board9, BLACK))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                possibleMoves =
                {{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.5,0,1.5,1.5,1.5,1.5},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.5,0,0,1.5,1.5,1.5,1.5},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.5,0,0,0,0,0,0,0,0,0,1.5,1.5,1.5,1.5},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.5,0,0,0,0,0,0,0,0,0,0,1.5,1.5,1.5,1.5},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.5,0,1.5,0,0,0,0,1.5,1.5,1.5},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.5,0,0,1.5,0,0,1.5,0,1.5,1.5},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.5,0,1.5,0,0,0,1.5,0,1.5,1.5},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.5,0,0,0,1.5,0,1.5,1.5,0,1.5},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.5,0,0,1.5,0,0,1.5,1.5,0,1.5},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.5,0,0,0,0,1.5,1.5,1.5,1.5,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.5,0,0,0,1.5,0,1.5,1.5,1.5,0}};

                moves.clear();
                for(auto vec : finder.possibleMoves(board10, BLACK))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                possibleMoves =
                {{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.5,-1.5,0,0,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.5,0,-1.5,0,0,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.5,0,0,0,0,0,0,-1.5}};

                moves.clear();
                for(auto vec : finder.possibleMoves(board11, RED))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                possibleMoves =
                {{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.5,0,0,-1.5,0,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.5,0,0,0,-1.5,0,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.5,0,0,0,0,-1.5,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.5,0,0,0,0,0,-1.5,0}};

                moves.clear();
                for(auto vec : finder.possibleMoves(board12, RED))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                possibleMoves =
                {{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.5,-1.5,0,0,-1.5,0,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.5,-1.5,0,0,0,0,-1.5,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.5,0,0,0,0,0,0,0,-1.5,-1.5,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.5,0,0,0,0,0,0,-1.5,-1.5,0}};

                moves.clear();
                for(auto vec : finder.possibleMoves(board13, RED))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                possibleMoves =
                {{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.5,-1.5,0,0,0,0,0,0,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.5,0,0,0,0,0,0,-1.5},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.5,0,0,0,0,0,0,-1.5,0,0,0,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.5,0,0,0,0,0,-1.5,0,0,0,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.5,-1.5,0,0,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.5,0,-1.5,0,0}};

                moves.clear();
                for(auto vec : finder.possibleMoves(board14, RED))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                possibleMoves =
                {{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.5,0,0,-1.5,0,0,0,0,0,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.5,0,-1.5,0,0,0,0,0,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.5,0,0,0,0,-1.5,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.5,0,0,0,0,0,-1.5},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.5,0,0,0,0,-1.5,0,0,0,0,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.5,0,0,0,-1.5,0,0,0,0,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.5,0,0,-1.5,0,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.5,0,0,0,-1.5,0}};

                moves.clear();
                for(auto vec : finder.possibleMoves(board15, RED))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                possibleMoves =
                {{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.5,-1.5,0,-1.5,0,0,0,0,0,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.5,0,0,-1.5,0,0,0,0,-1.5,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.5,0,0,-1.5,0,0,0,0,0,-1.5},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.5,-1.5,0,0,0,-1.5,0,0,0,0,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.5,0,0,0,-1.5,0,0,-1.5,0,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.5,0,0,0,-1.5,0,0,0,-1.5,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.5,0,0,0,0,0,0,-1.5,-1.5,0,0,0,0,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.5,0,0,0,0,0,0,0,-1.5,-1.5,0,0,0,0,0}};

                moves.clear();
                for(auto vec : finder.possibleMoves(board16, RED))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                possibleMoves =
                {{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.5,-1.5,-1.5,-1.5,0,-1.5,-1.5,-1.5,-1.5,-1.5,-1.5,-1.5,-1.5},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.5,0,-1.5,-1.5,-1.5,0,-1.5,-1.5,-1.5,-1.5,-1.5,-1.5,-1.5,-1.5},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.5,0,-1.5,-1.5,0,-1.5,-1.5,-1.5,-1.5,-1.5,-1.5,-1.5,-1.5,-1.5},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.5,0,0,-1.5,-1.5,0,-1.5,-1.5,-1.5,-1.5,-1.5,-1.5,-1.5,-1.5,-1.5},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.5,0,0,-1.5,0,-1.5,-1.5,-1.5,-1.5,-1.5,-1.5,-1.5,-1.5,-1.5,-1.5},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.5,0,0,0,-1.5,0,-1.5,-1.5,-1.5,-1.5,-1.5,-1.5,-1.5,-1.5,-1.5,-1.5},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.5,0,0,0,0,-1.5,-1.5,-1.5,-1.5,-1.5,-1.5,-1.5,-1.5,-1.5,-1.5,-1.5}};

                moves.clear();
                for(auto vec : finder.possibleMoves(board17, RED))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                possibleMoves =
                {{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.5,-1.5,-1.5,-1.5,-1.5,-1.5,-1.5,0,0,0,0,-1.5},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.5,-1.5,-1.5,-1.5,-1.5,-1.5,0,-1.5,0,0,-1.5,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.5,-1.5,-1.5,-1.5,-1.5,-1.5,0,-1.5,0,0,0,-1.5},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.5,-1.5,-1.5,-1.5,-1.5,0,-1.5,-1.5,0,-1.5,0,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.5,-1.5,-1.5,-1.5,-1.5,0,-1.5,-1.5,0,0,-1.5,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.5,-1.5,-1.5,-1.5,0,-1.5,-1.5,-1.5,-1.5,0,0,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.5,-1.5,-1.5,-1.5,0,-1.5,-1.5,-1.5,0,-1.5,0,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.5,-1.5,-1.5,-1.5,0,-1.5,-1.5,-1.5,-1.5,0,0,0,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.5,0,-1.5,-1.5,-1.5,0,-1.5,-1.5,-1.5,-1.5,0,0,0,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.5,0,-1.5,-1.5,0,-1.5,-1.5,-1.5,-1.5,-1.5,0,0,0,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.5,0,0,-1.5,-1.5,0,-1.5,-1.5,-1.5,-1.5,-1.5,0,0,0,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.5,0,0,-1.5,0,-1.5,-1.5,-1.5,-1.5,-1.5,-1.5,0,0,0,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.5,0,0,0,-1.5,0,-1.5,-1.5,-1.5,-1.5,-1.5,-1.5,0,0,0,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.5,0,0,0,0,-1.5,-1.5,-1.5,-1.5,-1.5,-1.5,-1.5,0,0,0,0}};

                moves.clear();
                for(auto vec : finder.possibleMoves(board18, RED))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                possibleMoves =
                {{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.5,-1.5,-1.5,-1.5,-1.5,-1.5,-1.5,0,0,0,0,-1.5,0,0,0,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.5,-1.5,-1.5,-1.5,-1.5,-1.5,-1.5,0,0,0,-1.5,0,0,0,0,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.5,-1.5,-1.5,-1.5,-1.5,-1.5,0,-1.5,0,0,-1.5,0,0,0,0,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.5,-1.5,-1.5,-1.5,-1.5,-1.5,0,-1.5,0,-1.5,0,0,0,0,0,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.5,-1.5,-1.5,-1.5,-1.5,0,-1.5,-1.5,0,-1.5,0,0,0,0,0,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.5,-1.5,-1.5,-1.5,-1.5,0,-1.5,-1.5,-1.5,0,0,0,0,0,0,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.5,-1.5,-1.5,-1.5,0,-1.5,-1.5,-1.5,-1.5,0,0,0,0,0,0,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.5,-1.5,-1.5,-1.5,0,-1.5,-1.5,-1.5,-1.5,0,0,0,0,0,0,0,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.5,0,-1.5,-1.5,0,-1.5,-1.5,-1.5,-1.5,-1.5,0,0,0,0,0,0,0,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.5,-1.5,-1.5,0,-1.5,-1.5,-1.5,-1.5,-1.5,0,0,0,0,0,0,0,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,-1.5,0,0,-1.5,0,-1.5,-1.5,-1.5,-1.5,-1.5,-1.5,0,0,0,0,0,0,0,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.5,0,-1.5,0,-1.5,-1.5,-1.5,-1.5,-1.5,-1.5,0,0,0,0,0,0,0,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,-1.5,0,0,0,0,-1.5,-1.5,-1.5,-1.5,-1.5,-1.5,-1.5,0,0,0,0,0,0,0,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,-1.5,0,0,0,-1.5,-1.5,-1.5,-1.5,-1.5,-1.5,-1.5,0,0,0,0,0,0,0,0}};

                moves.clear();
                for(auto vec : finder.possibleMoves(board19, RED))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                possibleMoves =
                {{-1.5,-1.5,-1.5,-1.5,0,-1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {-1.5,-1.5,-1.5,-1.5,0,0,-1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {-1.5,-1.5,-1.5,-1.5,0,0,0,0,0,0,0,0,0,-1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {-1.5,-1.5,-1.5,-1.5,0,0,0,0,0,0,0,0,0,0,-1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {-1.5,-1.5,-1.5,0,0,0,0,-1.5,0,-1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {-1.5,-1.5,0,-1.5,0,0,-1.5,0,0,-1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {-1.5,-1.5,0,-1.5,0,0,0,-1.5,0,-1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {-1.5,0,-1.5,-1.5,0,-1.5,0,0,0,-1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {-1.5,0,-1.5,-1.5,0,0,-1.5,0,0,-1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,-1.5,-1.5,-1.5,-1.5,0,0,0,0,-1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,-1.5,-1.5,-1.5,0,-1.5,0,0,0,-1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}};

                moves.clear();
                for(auto vec : finder.possibleMoves(board20, RED))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

            } // Then they should be able to move backwards
        } // When the board contains kings

        WHEN("Enemy pieces are present")
        {
            Board board1 =
            {0,0,0,0,0,0,0,0,0,1.5,0,0,0,0,0,0,-1.5,-1.5,-1.5,0,0,0,0,0,0,0,0,0,0,0,0,0};

            Board board2 =
            {0,0,0,0,0,0,0,0,-1.5,-1.5,0,0,0,1.5,0,0,-1.5,-1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

            Board board3 =
            {0,0,0,0,0,0,0,0,0,-1.5,0,0,0,1.5,1.5,1.5,-1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

            Board board4 =
            {0,0,0,0,0,0,0,0,0,0,0,0,-1.5,-1.5,0,0,1.5,1.5,1.5,0,-1.5,-1.5,0,0,0,0,0,0,0,0,0,0};

            Board board5 =
            {0,0,0,0,0,0,0,0,0,0,-1.5,-1.5,0,1.5,1.5,1.5,0,0,-1.5,-1.5,0,0,0,0,0,0,0,0,0,0,0,0};

            Board board6 =
            {-1.5,-1.5,1.5,0,0,1.5,-1.5,0,-1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

            Board board7 =
            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.5,1.5,0,0,0,-1.5,-1.5,1.5};

            Board board8 =
            {0,0,0,0,0,0,0,0,0,0,0,0,0,1.5,1.5,1.5,0,0,0,0,0,0,-1.5,0,0,0,0,0,0,0,0,0};

            Board board9 =
            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.5,1.5,0,0,-1.5,0,0,0,1.5,1.5,0,0,0,0,0,0,0,0};

            Board board10 =
            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.5,-1.5,-1.5,-1.5,0,0,0,1.5,0,0,0,0,0,0,0,0,0};

            Board board11 =
            {0,0,0,0,0,0,0,0,0,0,1.5,1.5,0,-1.5,-1.5,-1.5,0,0,1.5,1.5,0,0,0,0,0,0,0,0,0,0,0,0};

            Board board12 =
            {0,0,0,0,0,0,0,0,0,0,0,0,1.5,1.5,0,0,-1.5,-1.5,-1.5,0,1.5,1.5,0,0,0,0,0,0,0,0,0,0};

            Board board13 =
            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.5,0,1.5,-1.5,0,0,-1.5,1.5,1.5};

            Board board14 =
            {-1.5,1.5,1.5,0,0,0,-1.5,-1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

            THEN("Kings should jump backwards, and always jump when possible")
            {
                set<Board> possibleMoves =
                {{0,0,0,0,0,0,0,0,0,0,0,0,0,1.5,0,0,-1.5,-1.5,-1.5,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.5,0,-1.5,-1.5,-1.5,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,0,0,0,0,1.5,0,0,0,0,0,0,0,0,0,0,-1.5,-1.5,-1.5,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,0,0,0,0,0,1.5,0,0,0,0,0,0,0,0,0,-1.5,-1.5,-1.5,0,0,0,0,0,0,0,0,0,0,0,0,0}};

                set<Board> moves;
                for(auto vec : finder.possibleMoves(board1, BLACK))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                possibleMoves =
                {{0,0,0,0,0,0,0,0,-1.5,-1.5,0,0,0,0,0,0,0,-1.5,0,0,1.5,0,0,0,0,0,0,0,0,0,0,0},
                 {0,0,0,0,0,0,0,0,-1.5,-1.5,0,0,0,0,0,0,-1.5,0,0,0,0,0,1.5,0,0,0,0,0,0,0,0,0},
                 {0,0,0,0,1.5,0,0,0,0,-1.5,0,0,0,0,0,0,-1.5,-1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,0,0,0,0,0,1.5,0,-1.5,0,0,0,0,0,0,0,-1.5,-1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0}};

                moves.clear();
                for(auto vec : finder.possibleMoves(board2, BLACK))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                possibleMoves =
                {{0,0,0,0,0,0,0,0,0,-1.5,0,0,0,0,1.5,1.5,0,0,0,0,1.5,0,0,0,0,0,0,0,0,0,0,0},
                 {0,0,0,0,0,0,1.5,0,0,0,0,0,0,0,1.5,1.5,-1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,0,0,0,0,1.5,0,0,0,0,0,0,0,1.5,0,1.5,-1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}};

                moves.clear();
                for(auto vec : finder.possibleMoves(board3, BLACK))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                possibleMoves =
                {{0,0,0,0,0,0,0,0,0,0,0,0,-1.5,-1.5,0,0,0,1.5,1.5,0,-1.5,0,0,0,0,1.5,0,0,0,0,0,0},
                 {0,0,0,0,0,0,0,0,0,1.5,0,0,-1.5,0,0,0,0,1.5,1.5,0,-1.5,-1.5,0,0,0,0,0,0,0,0,0,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,-1.5,-1.5,0,0,1.5,0,1.5,0,-1.5,0,0,0,1.5,0,0,0,0,0,0,0},
                 {0,0,0,0,0,0,0,0,1.5,0,0,0,-1.5,0,0,0,1.5,0,1.5,0,-1.5,-1.5,0,0,0,0,0,0,0,0,0,0}};

                moves.clear();
                for(auto vec : finder.possibleMoves(board4, BLACK))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                possibleMoves =
                {{0,0,0,0,0,0,0,0,0,0,-1.5,-1.5,0,1.5,0,1.5,0,0,0,-1.5,0,0,0,1.5,0,0,0,0,0,0,0,0},
                 {0,0,0,0,0,0,0,1.5,0,0,0,-1.5,0,1.5,0,1.5,0,0,-1.5,-1.5,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,0,0,0,0,0,0,0,0,0,-1.5,-1.5,0,1.5,1.5,0,0,0,0,-1.5,0,0,1.5,0,0,0,0,0,0,0,0,0},
                 {0,0,0,0,0,0,1.5,0,0,0,0,-1.5,0,1.5,1.5,0,0,0,-1.5,-1.5,0,0,0,0,0,0,0,0,0,0,0,0}};

                moves.clear();
                for(auto vec : finder.possibleMoves(board5, BLACK))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                possibleMoves =
                {{-1.5,-1.5,0,0,0,1.5,0,0,-1.5,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {-1.5,-1.5,1.5,0,0,0,-1.5,0,0,0,0,0,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}};

                moves.clear();
                for(auto vec : finder.possibleMoves(board6, BLACK))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                possibleMoves =
                {{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.5,0,0,1.5,-1.5,-1.5,1.5},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.5,0,0,0,0,1.5,0,0,0,-1.5,-1.5,1.5},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.5,0,0,0,1.5,0,0,0,-1.5,-1.5,1.5},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.5,0,0,1.5,0,0,0,0,-1.5,-1.5,1.5},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.5,0,1.5,0,0,0,0,-1.5,-1.5,1.5},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.5,1.5,0,1.5,0,-1.5,-1.5,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.5,1.5,1.5,0,0,-1.5,-1.5,0}};

                moves.clear();
                for(auto vec : finder.possibleMoves(board7, BLACK))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                possibleMoves =
                {{0,0,0,0,0,0,0,0,0,0,0,0,0,1.5,1.5,1.5,0,0,-1.5,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,1.5,1.5,1.5,0,-1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,1.5,1.5,1.5,0,0,0,0,0,0,0,0,0,0,-1.5,0,0,0,0,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,1.5,1.5,1.5,0,0,0,0,0,0,0,0,0,-1.5,0,0,0,0,0,0}};

                moves.clear();
                for(auto vec : finder.possibleMoves(board8, RED))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                possibleMoves =
                {{0,0,0,0,0,0,0,0,0,0,0,-1.5,0,0,1.5,0,0,0,0,0,0,0,1.5,1.5,0,0,0,0,0,0,0,0},
                 {0,0,0,0,0,0,0,0,0,-1.5,0,0,0,0,0,1.5,0,0,0,0,0,0,1.5,1.5,0,0,0,0,0,0,0,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.5,1.5,0,0,0,0,0,0,1.5,0,0,0,0,-1.5,0,0,0,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.5,1.5,0,0,0,0,0,0,0,1.5,0,-1.5,0,0,0,0,0,0}};

                moves.clear();
                for(auto vec : finder.possibleMoves(board9, RED))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                possibleMoves =
                {{0,0,0,0,0,0,0,0,0,0,0,-1.5,0,0,0,0,-1.5,-1.5,0,0,0,0,1.5,0,0,0,0,0,0,0,0,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.5,-1.5,-1.5,0,0,0,0,0,0,0,-1.5,0,0,0,0,0,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.5,-1.5,0,-1.5,0,0,0,0,0,0,0,-1.5,0,0,0,0,0}};

                moves.clear();
                for(auto vec : finder.possibleMoves(board10, RED))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                possibleMoves =
                {{0,0,0,0,0,0,-1.5,0,0,0,0,1.5,0,-1.5,-1.5,0,0,0,1.5,1.5,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,0,0,0,0,0,0,0,0,0,1.5,1.5,0,-1.5,-1.5,0,0,0,0,1.5,0,0,-1.5,0,0,0,0,0,0,0,0,0},
                 {0,0,0,0,0,0,0,-1.5,0,0,0,1.5,0,-1.5,0,-1.5,0,0,1.5,1.5,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,0,0,0,0,0,0,0,0,0,1.5,1.5,0,-1.5,0,-1.5,0,0,0,1.5,0,0,0,-1.5,0,0,0,0,0,0,0,0}};

                moves.clear();
                for(auto vec : finder.possibleMoves(board11, RED))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                possibleMoves =
                {{0,0,0,0,0,0,0,0,-1.5,0,0,0,1.5,0,0,0,-1.5,0,-1.5,0,1.5,1.5,0,0,0,0,0,0,0,0,0,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,1.5,1.5,0,0,-1.5,0,-1.5,0,1.5,0,0,0,-1.5,0,0,0,0,0,0,0},
                 {0,0,0,0,0,0,0,0,0,-1.5,0,0,1.5,0,0,0,0,-1.5,-1.5,0,1.5,1.5,0,0,0,0,0,0,0,0,0,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,1.5,1.5,0,0,0,-1.5,-1.5,0,1.5,0,0,0,0,-1.5,0,0,0,0,0,0}};

                moves.clear();
                for(auto vec : finder.possibleMoves(board12, RED))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                possibleMoves =
                {{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.5,1.5,0,0,-1.5,0,0,0,1.5,1.5},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.5,0,0,0,0,0,1.5,0,0,0,-1.5,1.5,1.5}};

                moves.clear();
                for(auto vec : finder.possibleMoves(board13, RED))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                possibleMoves =
                {{-1.5,1.5,1.5,-1.5,0,0,-1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {-1.5,1.5,1.5,0,0,0,-1.5,0,0,0,0,-1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {-1.5,1.5,1.5,0,0,0,-1.5,0,0,0,-1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {-1.5,1.5,1.5,0,0,0,0,-1.5,0,0,-1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {-1.5,1.5,1.5,0,0,0,0,-1.5,0,-1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,1.5,1.5,0,-1.5,0,-1.5,-1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,1.5,1.5,0,0,-1.5,-1.5,-1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}};

                moves.clear();
                for(auto vec : finder.possibleMoves(board14, RED))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);
            } // Then kings should jump backwards
        } // When enemy pieces are present

        WHEN("Kings are all blocked")
        {
            Board board1 =
            {-1.5,0,0,0,1.5,0,0,0,-1.5,0,0,0,0,-1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

            Board board2 =
            {0,-1.5,0,0,-1.5,-1.5,0,0,1.5,0,0,0,-1.5,-1.5,0,0,0,-1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

            Board board3 =
            {0,0,-1.5,-1.5,0,0,0,1.5,0,0,-1.5,-1.5,0,0,-1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

            Board board4 =
            {0,0,-1.5,0,0,0,0,-1.5,0,0,0,1.5,0,0,0,-1.5,0,0,-1.5,0,0,0,0,0,0,0,0,0,0,0,0,0};

            Board board5 =
            {-1.5,0,-1.5,0,0,-1.5,-1.5,0,0,1.5,0,0,0,-1.5,-1.5,0,-1.5,0,-1.5,0,0,0,0,0,0,0,0,0,0,0,0,0};

            Board board6 =
            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.5,0,-1.5,0,-1.5,-1.5,-1.5,-1.5,0,1.5,0,1.5};

            Board board7 =
            {1.5,0,0,0,-1.5,-1.5,0,0,0,-1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

            Board board8 =
            {0,0,0,1.5,0,0,0,-1.5,0,0,-1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

            Board board9 =
            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.5,0,0,0,0,1.5,0,0,0,-1.5,0,0,0,1.5};

            Board board10 =
            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.5,0,0,0,1.5,1.5,0,0,0,-1.5,0,0,1.5,1.5,0,0,1.5,0};

            Board board11 =
            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.5,0,0,1.5,1.5,0,0,-1.5,0,0,0,1.5,1.5,0,0};

            Board board12 =
            {0,0,0,0,0,0,0,0,0,0,0,0,0,1.5,0,0,1.5,0,0,0,-1.5,0,0,0,1.5,0,0,0,0,1.5,0,0};

            Board board13 =
            {0,0,0,0,0,0,0,0,0,0,0,0,0,1.5,0,1.5,0,1.5,1.5,0,0,0,-1.5,0,0,1.5,1.5,0,0,1.5,0,1.5};

            Board board14 =
            {-1.5,0,-1.5,0,1.5,1.5,1.5,1.5,0,1.5,0,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

            Board board15 =
            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.5,0,0,0,1.5,1.5,0,0,0,-1.5};

            Board board16 =
            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.5,0,0,1.5,0,0,0,-1.5,0,0,0};

            THEN("They should not move")
            {
                set<Board> possibleMoves;

                set<Board> moves;
                for(auto vec : finder.possibleMoves(board1, BLACK))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                moves.clear();
                for(auto vec : finder.possibleMoves(board2, BLACK))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                moves.clear();
                for(auto vec : finder.possibleMoves(board3, BLACK))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                moves.clear();
                for(auto vec : finder.possibleMoves(board4, BLACK))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                moves.clear();
                for(auto vec : finder.possibleMoves(board5, BLACK))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                moves.clear();
                for(auto vec : finder.possibleMoves(board6, BLACK))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                moves.clear();
                for(auto vec : finder.possibleMoves(board7, BLACK))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                moves.clear();
                for(auto vec : finder.possibleMoves(board8, BLACK))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                moves.clear();
                for(auto vec : finder.possibleMoves(board9, RED))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                moves.clear();
                for(auto vec : finder.possibleMoves(board10, RED))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                moves.clear();
                for(auto vec : finder.possibleMoves(board11, RED))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                moves.clear();
                for(auto vec : finder.possibleMoves(board12, RED))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                moves.clear();
                for(auto vec : finder.possibleMoves(board13, RED))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                moves.clear();
                for(auto vec : finder.possibleMoves(board14, RED))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                moves.clear();
                for(auto vec : finder.possibleMoves(board15, RED))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                moves.clear();
                for(auto vec : finder.possibleMoves(board16, RED))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);
            } // Then they should not move
        } // When kings are all blocked

        WHEN("Jump sequences are available")
        {
            Board multijumpBlack =
            {0,0,0,0,0,1,0,0,0,-1,-1.5,0,0,0,0,1.5,0,-1.5,-1,0,0,0,0,0,0,0,0,0,0,0,0,0};

            Board multijumpRed =
            {0,0,0,0,0,0,0,0,0,0,0,0,0,1,1.5,0,-1.5,0,0,0,0,1.5,1,0,0,0,-1,0,0,0,0,0};

            THEN("They should be followed to their end")
            {
                set<Board> possibleMoves =
                {{0,0,0,0,0,0,0,0,0,0,-1.5,0,0,0,0,1.5,0,0,-1,0,0,1,0,0,0,0,0,0,0,0,0,0},
                 {0,0,0,0,0,0,0,0,0,0,-1.5,0,0,0,0,1.5,0,-1.5,0,0,0,0,0,1,0,0,0,0,0,0,0,0},
                 {0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}};

                set<Board> moves;
                for(auto vec : finder.possibleMoves(multijumpBlack, BLACK))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                possibleMoves =
                {{0,0,0,0,0,0,0,0,0,0,-1,0,0,1,0,0,-1.5,0,0,0,0,1.5,0,0,0,0,0,0,0,0,0,0},
                 {0,0,0,0,0,0,0,0,-1,0,0,0,0,0,1.5,0,-1.5,0,0,0,0,1.5,0,0,0,0,0,0,0,0,0,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.5,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.5,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0}};

                moves.clear();
                for(auto vec : finder.possibleMoves(multijumpRed, RED))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);
            }
        } // When Jump sequences are available

        WHEN("A pawn reaches the last row")
        {
            Board moveBlack =
            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,1,0,0,0,0,0};

            Board jumpBlack =
            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,-1,-1,0,0,0,0,0};

            Board moveRed =
            {0,0,0,0,0,-1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

            Board jumpRed =
            {0,0,0,0,0,1,1,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

            THEN("It should end the turn")
            {
                set<Board> possibleMoves =
                {{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,1.5,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,1.5}};

                set<Board> moves;
                for(auto vec : finder.possibleMoves(moveBlack, BLACK))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                possibleMoves =
                {{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,1.5,0}};

                moves.clear();
                for(auto vec : finder.possibleMoves(jumpBlack, BLACK))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                possibleMoves =
                {{0,-1.5,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {-1.5,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}};

                moves.clear();
                for(auto vec : finder.possibleMoves(moveRed, RED))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                possibleMoves =
                {{0,-1.5,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}};

                moves.clear();
                for(auto vec : finder.possibleMoves(jumpRed, RED))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);
            } // It should end the turn
        } // When a pawn reaches the last row

        WHEN("A king jumps to the last row")
        {
            Board moveBlack =
            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.5,1.5,0,0,0,0,0};

            Board jumpBlack =
            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.5,0,-1.5,-1.5,0,0,0,0,0};

            Board moveRed =
            {0,0,0,0,0,-1.5,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

            Board jumpRed =
            {0,0,0,0,0,1.5,1.5,0,-1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

            THEN("It should not end the turn")
            {
                set<Board> possibleMoves =
                {{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.5,0,0,0,0,1.5,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.5,0,0,0,0,0,1.5},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.5,0,0,-1.5,0,0,0,0,0,0},
                 {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.5,0,-1.5,0,0,0,0,0,0}};

                set<Board> moves;
                for(auto vec : finder.possibleMoves(moveBlack, BLACK))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                possibleMoves =
                {{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.5,0,0,0,0,0,0,0,0,0,0}};

                moves.clear();
                for(auto vec : finder.possibleMoves(jumpBlack, BLACK))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                possibleMoves =
                {{0,-1.5,0,0,0,0,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {-1.5,0,0,0,0,0,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,0,0,0,0,0,1.5,0,0,-1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                 {0,0,0,0,0,0,1.5,0,-1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}};

                moves.clear();
                for(auto vec : finder.possibleMoves(moveRed, RED))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);

                possibleMoves =
                {{0,0,0,0,0,0,0,0,0,0,-1.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}};

                moves.clear();
                for(auto vec : finder.possibleMoves(jumpRed, RED))
                {
                    moves.insert(vec);
                }

                REQUIRE(moves == possibleMoves);
            } // It shouldn't end the turn
        } // When a king reaches the last row
    } // Given a CheckersMoveFinder
}



float minimize(vector<float>& board, float beta, size_t depth, unsigned max_depth,
               bool player, shared_ptr<NeuralNet> nn, shared_ptr<MoveFinder> finder );

// board must have 32 values
// depth is >= 1
float
maximize( vector<float>& board, float beta, size_t depth, unsigned max_depth,
          bool player, shared_ptr<NeuralNet> nn, shared_ptr<MoveFinder> finder )
{
    if( depth < max_depth )
    {
        vector<vector<float>> moves = finder->possibleMoves(board, player);
        float alpha = MIN_FLOAT;

        for( auto ii = 0u; ii < moves.size(); ++ii )
        {
            alpha = std::max( alpha, minimize( moves[ii], alpha, depth + 1,
                              max_depth, player, nn, finder) );
        }

        return alpha;
    }
    else    //terminal node
    {
        return nn->evaluate( board );
    }
}

// board must have 32 values
// depth is >= 1
float
minimize( vector<float>& board, float beta, size_t depth, unsigned max_depth,
          bool player, shared_ptr<NeuralNet> nn, shared_ptr<MoveFinder> finder )
{
    if( depth < max_depth )
    {
        vector<vector<float>> moves = finder->possibleMoves(board, !player);
        float alpha = MAX_FLOAT;

        for( auto ii = 0u; ii < moves.size(); ++ii )
        {
            alpha = std::min( alpha, maximize( moves[ii], alpha, depth + 1,
                              max_depth, player, nn, finder) );
        }

        return alpha;
    }
    else    // terminal node
    {
        return nn->evaluate( board );
    }
}

vector<float>
miniMax(const vector<float>& current_state, unsigned max_depth,
        bool player, shared_ptr<NeuralNet> nn, shared_ptr<MoveFinder> finder)
{
    vector<vector<float>> moves = finder->possibleMoves(current_state, player);
    vector<float> best_move = current_state;
    float alpha = MIN_FLOAT;

    // Find which possible move has the highest guaranteed value
    for( auto ii = 0u; ii < moves.size(); ++ii )
    {
        float value = minimize( moves[ii], alpha, 1,
                                max_depth, player, nn, finder);
        if( value >= alpha )    // If it is better than the previous best, save it
        {
            alpha = value;
            best_move = moves[ii];
        }
    }

    return best_move;
}

SCENARIO("AB_Search tree for checkers", "absearch")
{
    const bool BLACK = true;
    const bool RED = false;

    GIVEN("An AB_Search object, a CheckersNet, and a CheckersMoveFinder")
    {
        vector<size_t> node_structure = {32, 128, 66, 25, 48, 1};
        shared_ptr<CheckersNet> nn = make_shared<CheckersNet>(
                node_structure, make_shared<SigmoidSynapse>(1));

        shared_ptr<MoveFinder> mf = make_shared<CheckersMoveFinder>();

        AB_Search blackSearch(nn, mf, 4, BLACK);
        AB_Search redSearch(nn, mf, 4, RED);

        WHEN("The AB search is run with a given game state")
        {
            // Generate a bunch of random boards
            srand(time(0));
            vector<vector<float>> boards;

            for(auto ii = 0u; ii < 100; ++ii)
            {
                vector<float> board;

                for(auto ii = 0u; ii < 32; ++ii)
                {
                    float piece = 0;

                    if( rand() % 2 ) // 50% chance
                    {
                        piece = (rand() % 2) * 2 - 1; // -1 or 1

                        if( !(rand() % 4) ) // 25% chance
                        {
                            piece *= 1.5;
                        }
                    }

                    board.push_back(piece);
                }

                boards.push_back(board);
            }

            for(auto ii = 0; ii < 10; ++ii)
            {
                nn->player(CheckersNet::BLACK);
                vector<float> ab_black = blackSearch(boards[ii]);
                vector<float> mini_max_black = miniMax(boards[ii], 4, BLACK, nn, mf);

                nn->player(CheckersNet::RED);
                vector<float> ab_red = redSearch(boards[ii]);
                vector<float> mini_max_red = miniMax(boards[ii], 4, RED, nn, mf);

                THEN("It should return the same move as a standard mini-max search")
                {
                    REQUIRE(ab_black == mini_max_black);
                    REQUIRE(ab_red == mini_max_red);
                }

                THEN("It should not return an empty vector")
                {
                    REQUIRE(ab_black != vector<float>());
                    REQUIRE(ab_red != vector<float>());
                }
            }

            THEN("It should run no slower than a standard mini-max search")
            {
                typedef high_resolution_clock clock;

                clock::time_point start_time = clock::now();
                vector<float> result = blackSearch(boards[0]);
                clock::time_point end_time   = clock::now();
                duration<double> ab_time = duration_cast<duration<double>>(end_time - start_time);

                start_time = clock::now();
                vector<float> mini_max_black = miniMax(boards[0], 4, BLACK, nn, mf);
                end_time   = clock::now();
                duration<double> mm_time = duration_cast<duration<double>>(end_time - start_time);

                REQUIRE(ab_time <= mm_time);
            }
        }

        WHEN("The AB search is run with an empty board")
        {
            vector<float> empty_board(32, 0);

            THEN("It should return an empty board")
            {
                nn->player(CheckersNet::BLACK);
                vector<float> ab_result = blackSearch(empty_board);

                REQUIRE(ab_result == empty_board);

                nn->player(CheckersNet::RED);
                ab_result = redSearch(empty_board);

                REQUIRE(ab_result == empty_board);
            }
        }

        WHEN("The AB search is run with a board with all moves blocked")
        {
            vector<float> board =
            {0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,0,0,0,0,0,0,0,0};

            THEN("It should return the original board")
            {
                nn->player(CheckersNet::BLACK);
                vector<float> ab_result = blackSearch(board);

                REQUIRE(ab_result == board);

                nn->player(CheckersNet::RED);
                ab_result = redSearch(board);

                REQUIRE(ab_result == board);
            }
        }

        WHEN("Given a board which will ultimately result in a loss")
        {
            vector<float> blackloses1 =
            {0,0,0,-1.5,0,0,-1,0,1,0,0,0,1,0,0,0,0,0,0,0,-1,-1,0,0,0,-1,0,1,0,0,-1,-1};

            vector<float> blackloses2 =
            {0,0,0,0,0,0,0,1,0,0,0,0,0,0,-1.5,0,0,-1,-1,0,0,0,0,0,0,-1,0,0,-1,0,0,0};

            vector<float> redloses1 =
            {1,1,0,0,-1,0,1,0,0,0,1,1,0,0,0,0,0,0,0,-1,0,0,0,-1,0,1,0,0,1.5,0,0,0,};

            vector<float> redloses2 =
            {0,0,0,1,0,0,1,0,0,0,0,0,0,1,1,0,0,1.5,0,0,0,0,0,0,-1,0,0,0,0,0,0,0};

            THEN("The AB_Search should not return the current board")
            {
                nn->player(CheckersNet::BLACK);

                vector<float> ab_result = blackSearch(blackloses1);
                REQUIRE(ab_result != blackloses1);

                ab_result = blackSearch(blackloses2);
                REQUIRE(ab_result != blackloses2);

                nn->player(CheckersNet::RED);

                ab_result = redSearch(redloses1);
                REQUIRE(ab_result != redloses1);

                ab_result = redSearch(redloses2);
                REQUIRE(ab_result != redloses2);
            }
        }
    } // Given an AB_Search object, a CheckersNet, and a CheckersMoveFinder
} // AB search tree for checkers

// TODO: Write tests for Generator
