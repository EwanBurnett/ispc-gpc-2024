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

#define PICOBENCH_IMPLEMENT_WITH_MAIN
#define PICOBENCH_DEFAULT_ITERATIONS {4096, 8192, 16384, 32768}
#include "picobench/picobench.hpp"

#include "part_2.h"
#include "part_2_ispc.h"
#include "vector3.h"

using std::vector;

namespace
{
	using Types::Vector3;
	using Types::Vector3_SSE;

	// we're using static seeds so we get the same numbers every time
	static constexpr uint32_t RAND_SEED = 0xBAAABAAA;

	// get a random float
	static inline float GetRandFloat(std::mt19937& generator)
	{
		return (float)generator() / (float)generator.max();
	}


	// initialize an AoS vector
	void InitializeAoS(vector<Vector3>& vec, const size_t count)
	{
		std::mt19937 generator(RAND_SEED);

		vec.resize(count);

		for (size_t i = 0; i < count; ++i)
		{
			vec[i].x = GetRandFloat(generator);
			vec[i].y = GetRandFloat(generator);
			vec[i].z = GetRandFloat(generator);
		}
	}

	// initialize an aligned AoS vector
	void InitializeAoS_SSE(vector<Vector3_SSE>& vec, const size_t count)
	{
		vec.resize(count);

		// initialize temp with random data
		std::mt19937 generator(RAND_SEED);

		for (size_t i = 0; i < count; ++i)
		{
			vec[i].m_vec = _mm_setr_ps( GetRandFloat(generator), GetRandFloat(generator), GetRandFloat(generator), 1.0f);
		}
	}

	// initialize an SoA vector
	void InitializeSoA(std::vector<float>& x, std::vector<float>& y, std::vector<float>& z, const size_t count)
	{
		std::mt19937 generator(RAND_SEED);

		x.resize(count, 0.f);
		y.resize(count, 0.f);
		z.resize(count, 0.f);

		for (size_t i = 0; i < count; ++i)
		{
			x[i] = GetRandFloat(generator);
			y[i] = GetRandFloat(generator);
			z[i] = GetRandFloat(generator);
		}
	}

	// initialize an AoSoA vector
	void InitializeAoSoA(std::vector<float>& vec, const size_t count)
	{
		std::mt19937 generator(RAND_SEED);

		int programCount = ispc::GetProgramCount();

		/// resize the vector
		vec.resize(count * (programCount * 3));

		// iterate through the structures
		for (size_t v = 0; v < count; v += (programCount * 3))
		{
			for (size_t i = v; i < programCount; ++i)
			{
				vec[i] = GetRandFloat(generator);
				vec[i + programCount] = GetRandFloat(generator);
				vec[i + (programCount * 2)] = GetRandFloat(generator);
			}
		}
	}
}

static void dot_CPP_Serial(picobench::state& s)
{
	vector<float> output(0.0f, s.iterations());
	vector<Vector3> vec;

	output.resize(s.iterations());
	vec.reserve(s.iterations());

	InitializeAoS(vec, s.iterations());

	s.start_timer();

	#pragma loop(no_vector)
	for (int i = 0; i < s.iterations(); ++i)
	{
		DotProductCpp(output, vec, s.iterations());
		s.set_result((uintptr_t)&output);
	}

	s.stop_timer(); // Manual stop
}
PICOBENCH(dot_CPP_Serial);

static void dot_CPP_vhaddps(picobench::state& s)
{
	vector<float> output(0.0f, s.iterations());
	vector<Vector3_SSE> vec;

	output.resize(s.iterations());
	vec.reserve(s.iterations());

	InitializeAoS_SSE(vec, s.iterations());

	s.start_timer();

	#pragma loop(no_vector)
	for (int i = 0; i < s.iterations(); ++i)
	{
		Types::DotProduct_HADD(output, vec.data(), s.iterations());
		s.set_result((uintptr_t)&output);
	}

	s.stop_timer(); // Manual stop
}
PICOBENCH(dot_CPP_vhaddps);

static void dot_CPP_vddps(picobench::state& s)
{
	vector<float> output(0.0f, s.iterations());
	vector<Vector3_SSE> vec;

	output.resize(s.iterations());
	vec.reserve(s.iterations());

	InitializeAoS_SSE(vec, s.iterations());

	s.start_timer();

	#pragma loop(no_vector)
	for (int i = 0; i < s.iterations(); ++i)
	{
		Types::DotProduct_DPPS(output, vec.data(), s.iterations());
		s.set_result((uintptr_t)&output);
	}

	s.stop_timer(); // Manual stop
}
PICOBENCH(dot_CPP_vddps);

static void dot_CPP_vmul_shuffle_add(picobench::state& s)
{
	vector<float> output(0.0f, s.iterations());
	vector<Vector3_SSE> vec;

	output.resize(s.iterations());
	vec.reserve(s.iterations());

	InitializeAoS_SSE(vec, s.iterations());

	s.start_timer();

	#pragma loop(no_vector)
	for (int i = 0; i < s.iterations(); ++i)
	{
		Types::DotProduct_SHUFFLE(output, vec.data(), s.iterations());
		s.set_result((uintptr_t)&output);
	}

	s.stop_timer(); // Manual stop
}
PICOBENCH(dot_CPP_vmul_shuffle_add);


static void dot_ispc_AoS(picobench::state& s)
{
	vector<float> output(0.0f, s.iterations());
	vector<Vector3> vec;

	output.resize(s.iterations());
	vec.reserve(s.iterations());

	InitializeAoS(vec, s.iterations());

	s.start_timer();

	#pragma loop(no_vector)
	for (int i = 0; i < s.iterations(); ++i)
	{
		ispc::DotProductAoS(output.data(), (ispc::Vector3*) vec.data(), s.iterations());
		s.set_result((uintptr_t)&output);
	}

	s.stop_timer(); // Manual stop
}
PICOBENCH(dot_ispc_AoS);

static void dot_ispc_SoA(picobench::state& s)
{
	vector<float> output(0.0f, s.iterations());
	vector<float> x;
	vector<float> y;
	vector<float> z;

	output.resize(s.iterations());
	x.reserve(s.iterations());
	y.reserve(s.iterations());
	z.reserve(s.iterations());

	InitializeSoA(x, y, z, s.iterations());
	s.start_timer();

	#pragma loop(no_vector)
	for (int i = 0; i < s.iterations(); ++i)
	{
		ispc::DotProductSoA(output.data(), x.data(), y.data(), z.data(), s.iterations());
		s.set_result((uintptr_t)&output);
	}

	s.stop_timer(); // Manual stop
}
PICOBENCH(dot_ispc_SoA);

static void dot_ispc_AoSoA(picobench::state& s)
{
	vector<float> output(0.0f, s.iterations());
	vector<float> vec;

	output.resize(s.iterations());

	InitializeAoSoA(vec, s.iterations());
	s.start_timer();

	#pragma loop(no_vector)
	for (int i = 0; i < s.iterations(); ++i)
	{
		ispc::DotProductAoSoA(output.data(), vec.data(), s.iterations());
		s.set_result((uintptr_t)&output);
	}

	s.stop_timer(); // Manual stop
}
PICOBENCH(dot_ispc_AoSoA);