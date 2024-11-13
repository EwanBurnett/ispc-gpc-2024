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

export void AddArrayElements(uniform float output[], const uniform float a[], const uniform float b[], const uniform int64 count)
{
    foreach(i = 0 ... count)    //foreach is evil, but... it works!
    {
        output[i] = a[i] + b[i];
    }
}

export void SumArray(uniform float& sum_output, const uniform float a[], const uniform int64 count)
{
    varying float sum = 0;
    foreach(i = 0 ... count) //Accumulate the result for each lane
    {
        sum += a[i];
    }

    sum_output = reduce_add(sum);//Each lane has its own sum; add all of them together
}


export void MinArray(uniform float& min_output, const uniform float a[], const uniform int64 count)
{
    varying float minimum = FLT_MAX;

    foreach(i = 0 ... count)
    {
        if( a[i] < minimum )    //ISPC handles branching via masks internally!
        {
            minimum = a[i];
        }


    }

    min_output = reduce_min(minimum);//Find the minimum of each element in the resultant lane
}


export void MaxArray(uniform float& max_output, const uniform float a[], const uniform int64 count)
{
    varying float maximum = -FLT_MAX;

    foreach(i = 0 ... count)
    {
        if( a[i] > maximum )
        {
            maximum = a[i];
        }


    }

    max_output = reduce_max(maximum);
}


export void AverageArray(uniform float& avg_output, const uniform float a[], const uniform int64 count)
{

    varying float sum = 0;

    foreach(i = 0 ... count)
    {
        sum += a[i];
    }

    uniform float sum_reduce = reduce_add(sum);

    avg_output = sum_reduce / (uniform float) count;
}
