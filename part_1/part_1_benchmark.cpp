// Copyright(c) 2024, Pete Brubaker <pete.brubaker@intel.com>
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files(the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// ISPC: Making CPU SIMD fun while tracing rays!
// Part 1
//
// Graphics Programming Conference 2024
// https://www.graphicsprogrammingconference.nl/
//
// Simple Iteration and Reductions
//


#define PICOBENCH_IMPLEMENT_WITH_MAIN
#define PICOBENCH_DEFAULT_ITERATIONS {4096, 8192, 16384, 32768}
#include "picobench/picobench.hpp"

#include <vector>
#include <random>
#include <algorithm>

#include "part_1.h"
#include "part_1_ispc.h"

using std::vector;

// we're using static seeds so we get the same numbers every time
static constexpr uint32_t RAND_SEED_A = 0xBAAABAAA;
static constexpr uint32_t RAND_SEED_B = 0xB000B000;

namespace
{
	void InitializeArray(std::vector<float>& array, const size_t count)
	{
		std::mt19937 generator(RAND_SEED_A);
		std::uniform_real_distribution<float> distribution(0.0f, 10.0f);

		std::generate_n(std::back_inserter(array), count, [&] { return distribution(generator); });
	}

	void InitializeDoubleArray(std::vector<float>& arrayA, std::vector<float>& arrayB, const size_t count)
	{
		std::mt19937 generator_a(RAND_SEED_A);
		std::uniform_real_distribution<float> distribution_a(0.0f, 10.0f);

		std::generate_n(std::back_inserter(arrayA), count, [&] { return distribution_a(generator_a); });

		std::mt19937 generator_b(RAND_SEED_B);
		std::uniform_real_distribution<float> distribution_b(0.0f, 10.0f);

		std::generate_n(std::back_inserter(arrayB), count, [&] { return distribution_b(generator_b); });
	}
}

static void AddArrayElements_CPP(picobench::state& s)
{
	vector<float> output(0.0f, s.iterations());
	vector<float> a;
	vector<float> b;

	output.resize(s.iterations());
	a.reserve(s.iterations());
	b.reserve(s.iterations());

	InitializeDoubleArray(a, b, s.iterations());

    s.start_timer();
    
	#pragma loop(no_vector)
    for ( int i = 0; i < s.iterations(); ++i )
	{
        AddArrayElements(output, a, b, s.iterations());
	}

    s.stop_timer(); // Manual stop
}
PICOBENCH(AddArrayElements_CPP);

static void AddArrayElements_ISPC(picobench::state& s)
{
	vector<float> output(0.0f, s.iterations());
	vector<float> a;
	vector<float> b;

	output.resize(s.iterations());
	a.reserve(s.iterations());
	b.reserve(s.iterations());

	InitializeDoubleArray(a, b, s.iterations());

	s.start_timer();

	#pragma loop(no_vector)
	for (int i = 0; i < s.iterations(); ++i)
	{
		ispc::AddArrayElements(output.data(), a.data(), b.data(), s.iterations());
	}

	s.stop_timer(); // Manual stop
}
PICOBENCH(AddArrayElements_ISPC);


static void SumArray_CPP(picobench::state& s)
{
	float output = 0.0f;
	vector<float> a;

	a.reserve(s.iterations());

	InitializeArray(a, s.iterations());

    s.start_timer();
    
	#pragma loop(no_vector)
    for ( int i = 0; i < s.iterations(); ++i )
	{
        SumArray(output, a, s.iterations());

		s.set_result((uintptr_t)&output);
	}

    s.stop_timer(); // Manual stop
}
PICOBENCH(SumArray_CPP);

static void SumArray_ISPC(picobench::state& s)
{
	float output = 0.0f;
	vector<float> a;

	a.reserve(s.iterations());

	InitializeArray(a, s.iterations());

	s.start_timer();

	#pragma loop(no_vector)
	for (int i = 0; i < s.iterations(); ++i)
	{
		ispc::SumArray(output, a.data(), s.iterations());

		s.set_result((uintptr_t)&output);
	}

	s.stop_timer(); // Manual stop
}
PICOBENCH(SumArray_ISPC);


static void MinArray_CPP(picobench::state& s)
{
	float output = 0.0f;
	vector<float> a;

	a.reserve(s.iterations());

	InitializeArray(a, s.iterations());

	s.start_timer();

	#pragma loop(no_vector)
	for (int i = 0; i < s.iterations(); ++i)
	{
		MinArray(output, a, s.iterations());

		s.set_result((uintptr_t)&output);
	}

	s.stop_timer(); // Manual stop
}
PICOBENCH(MinArray_CPP);

static void MinArray_ISPC(picobench::state& s)
{
	float output = 0.0f;
	vector<float> a;

	a.reserve(s.iterations());

	InitializeArray(a, s.iterations());

	s.start_timer();

	#pragma loop(no_vector)
	for (int i = 0; i < s.iterations(); ++i)
	{
		ispc::MinArray(output, a.data(), s.iterations());

		s.set_result((uintptr_t)&output);
	}

	s.stop_timer(); // Manual stop
}
PICOBENCH(MinArray_ISPC);


static void MaxArray_CPP(picobench::state& s)
{
	float output = 0.0f;
	vector<float> a;

	a.reserve(s.iterations());

	InitializeArray(a, s.iterations());

	s.start_timer();

	#pragma loop(no_vector)
	for (int i = 0; i < s.iterations(); ++i)
	{
		MaxArray(output, a, s.iterations());

		s.set_result((uintptr_t)&output);
	}

	s.stop_timer(); // Manual stop
}
PICOBENCH(MaxArray_CPP);


static void MaxArray_ISPC(picobench::state& s)
{
	float output = 0.0f;
	vector<float> a;

	a.reserve(s.iterations());

	InitializeArray(a, s.iterations());

	s.start_timer();

	#pragma loop(no_vector)
	for (int i = 0; i < s.iterations(); ++i)
	{
		ispc::MaxArray(output, a.data(), s.iterations());

		s.set_result((uintptr_t)&output);
	}

	s.stop_timer(); // Manual stop
}
PICOBENCH(MaxArray_ISPC);


static void AverageArray_CPP(picobench::state& s)
{
	float output = 0.0f;
	vector<float> a;

	a.reserve(s.iterations());

	InitializeArray(a, s.iterations());

	s.start_timer();

	#pragma loop(no_vector)
	for (int i = 0; i < s.iterations(); ++i)
	{
		AverageArray(output, a, s.iterations());

		s.set_result((uintptr_t)&output);
	}

	s.stop_timer(); // Manual stop
}
PICOBENCH(AverageArray_CPP);


static void AverageArray_ISPC(picobench::state& s)
{
	float output = 0.0f;
	vector<float> a;

	a.reserve(s.iterations());

	InitializeArray(a, s.iterations());

	s.start_timer();

	#pragma loop(no_vector)
	for (int i = 0; i < s.iterations(); ++i)
	{
		ispc::AverageArray(output, a.data(), s.iterations());

		s.set_result((uintptr_t)&output);
	}

	s.stop_timer(); // Manual stop
}
PICOBENCH(AverageArray_ISPC);