/**
 * Copyright (c) 2009/09-2012/08, Regents of the University of Colorado
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
/**
 * Copyright 2012/09-2013/04, 2013/11-Present, University of Massachusetts Amherst
 * Copyright 2013/05-2013/10, IPSoft Inc.
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. 
 */
package com.clearnlp.classification.algorithm;

import java.util.Arrays;

import com.clearnlp.classification.instance.IntInstance;
import com.clearnlp.classification.model.StringModelAD;

/**
 * Abstract algorithm.
 * @since 1.3.2
 * @author Jinho D. Choi ({@code jdchoi77@gmail.com})
 */
abstract public class AbstractAdaGrad extends AbstractAlgorithm
{
	protected double[] d_gradients;
	protected double[] d_average;
	protected boolean  b_average;
	protected double   d_alpha;
	protected double   d_rho;
	
	abstract protected boolean update(StringModelAD model, IntInstance instance, int averageCount);
	
	public AbstractAdaGrad(double alpha, double rho, boolean average)
	{
		super(LEARN_ONLINE);
		init(alpha, rho, average);
	}
	
	public void init(double alpha, double rho, boolean average)
	{
		d_alpha   = alpha;
		d_rho     = rho;
		b_average = average;
	}
	
	@Override
	public void train(StringModelAD model)
	{	
		final int LD = model.getLabelSize() * model.getFeatureSize();
		final int N  = model.getInstanceSize();
		
		model.shuffleIndices();
		
		if (d_gradients == null || d_gradients.length != LD)
		{
			d_gradients = new double[LD];
			if (b_average) d_average = new double[LD];
		}
		else
		{
			Arrays.fill(d_gradients, 0d);
			if (b_average) Arrays.fill(d_average, 0d);
		}
		
		int i; for (i=0; i<N; i++)
			update(model, model.getInstance(model.getShuffledIndex(i)), i+1);
		
		if (b_average) 
			model.setAverageWeights(d_average, N+1);
	}
	
	protected void updateWeight(StringModelAD model, int y, int x, double v, int averageCount)
	{
		double cost = getCost(model, y, x) * v;
		model.updateWeight(y, x, (float)cost);
		if (b_average) d_average[model.getWeightIndex(y,x)] += cost * averageCount;
	}
	
	protected double getCost(StringModelAD model, int y, int x)
	{
		return d_alpha / (d_rho + Math.sqrt(d_gradients[model.getWeightIndex(y, x)]));
	}
}
