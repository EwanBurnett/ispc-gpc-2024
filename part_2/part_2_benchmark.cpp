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

static void dot_CPP(picobench::state& s)
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
	}

	s.stop_timer(); // Manual stop
}
PICOBENCH(dot_CPP);

/*

BENCHMARK_DEFINE_F(Vector3_SSE, vhaddps)(benchmark::State& st)
{
	size_t array_size = (size_t)st.range(0);

	for (auto _ : st)
	{
		DotProduct_HADD(m_squared_length, m_v, array_size);
		benchmark::DoNotOptimize(m_squared_length);
	}
}

BENCHMARK_DEFINE_F(Vector3_SSE, vdpps)(benchmark::State& st)
{
	size_t array_size = (size_t)st.range(0);

	for (auto _ : st)
	{
		DotProduct_DPPS(m_squared_length, m_v, array_size);
		benchmark::DoNotOptimize(m_squared_length);
	}
}

BENCHMARK_DEFINE_F(Vector3_SSE, vmul_shuffle_add)(benchmark::State& st)
{
	size_t array_size = (size_t)st.range(0);

	for (auto _ : st)
	{
		DotProduct_SHUFFLE(m_squared_length, m_v, array_size);
		benchmark::DoNotOptimize(m_squared_length);
	}
}

BENCHMARK_DEFINE_F(Vector3, ispc_AoS)(benchmark::State& st)
{
	size_t array_size = (size_t)st.range(0);

	for (auto _ : st)
	{
		ispc::DotProductAoS(m_squared_length.data(), (ispc::Vector3*) m_v.data(), array_size);
		benchmark::DoNotOptimize(m_squared_length);
	}
}

BENCHMARK_DEFINE_F(Vector3SoA, ispc_SoA)(benchmark::State& st)
{
	size_t array_size = (size_t)st.range(0);

	for (auto _ : st)
	{
		ispc::DotProductSoA(m_squared_length.data(), m_x.data(), m_y.data(), m_z.data(), array_size);
		benchmark::DoNotOptimize(m_squared_length);
	}
}

BENCHMARK_DEFINE_F(Vector3AoSoA, ispc_AoSoA)(benchmark::State& st)
{
	size_t array_size = (size_t)st.range(0);

	for (auto _ : st)
	{
		ispc::DotProductAoSoA(m_squared_length.data(), (ispc::Vector3AoSoA*)m_v.data(), array_size);
		benchmark::DoNotOptimize(m_squared_length);
	}
}

BENCHMARK_REGISTER_F(Vector3, Cpp)->RangeMultiplier(2)->Range(8 << 12, 8 << 23);
BENCHMARK_REGISTER_F(Vector3_SSE, vhaddps)->RangeMultiplier(2)->Range(8 << 12, 8 << 23);
BENCHMARK_REGISTER_F(Vector3_SSE, vdpps)->RangeMultiplier(2)->Range(8 << 12, 8 << 23);
BENCHMARK_REGISTER_F(Vector3_SSE, vmul_shuffle_add)->RangeMultiplier(2)->Range(8 << 12, 8 << 23);
BENCHMARK_REGISTER_F(Vector3, ispc_AoS)->RangeMultiplier(2)->Range(8 << 12, 8 << 23);
BENCHMARK_REGISTER_F(Vector3SoA, ispc_SoA)->RangeMultiplier(2)->Range(8 << 12, 8 << 23);
BENCHMARK_REGISTER_F(Vector3AoSoA, ispc_AoSoA)->RangeMultiplier(2)->Range(8 << 12, 8 << 23);

// using our own main allows us to do things like change the default time unit
int main(int argc, char** argv)
{
	::benchmark::Initialize(&argc, argv);

	::benchmark::SetDefaultTimeUnit(benchmark::kMillisecond);

	if (::benchmark::ReportUnrecognizedArguments(argc, argv))
		return 1;

	::benchmark::RunSpecifiedBenchmarks();

	_mm_pause();
}
*/
