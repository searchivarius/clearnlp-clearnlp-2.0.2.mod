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
package com.clearnlp.generation;

import static org.junit.Assert.assertEquals;

import java.io.File;
import java.util.zip.ZipFile;

import org.junit.Test;

import com.clearnlp.dictionary.DTLib;


/** @author Jinho D. Choi ({@code jdchoi77@gmail.com}) */
public class LGVerbEnTest
{
	@Test
	public void testGet3rdSingularForm()
	{
		String s = "work";
		assertEquals(LGVerbEn.get3rdSingularForm(s), "works");
		
		s = "hope";
		assertEquals(LGVerbEn.get3rdSingularForm(s), "hopes");
		
		s = "teach";
		assertEquals(LGVerbEn.get3rdSingularForm(s), "teaches");
		
		s = "wish";
		assertEquals(LGVerbEn.get3rdSingularForm(s), "wishes");
		
		s = "miss";
		assertEquals(LGVerbEn.get3rdSingularForm(s), "misses");
		
		s = "buzz";
		assertEquals(LGVerbEn.get3rdSingularForm(s), "buzzes");
		
		s = "fix";
		assertEquals(LGVerbEn.get3rdSingularForm(s), "fixes");
		
		s = "go";
		assertEquals(LGVerbEn.get3rdSingularForm(s), "goes");
		
		s = "stay";
		assertEquals(LGVerbEn.get3rdSingularForm(s), "stays");
		
		s = "enjoy";
		assertEquals(LGVerbEn.get3rdSingularForm(s), "enjoys");
		
		s = "fly";
		assertEquals(LGVerbEn.get3rdSingularForm(s), "flies");
	}
	
	@Test
	public void testGetPastRegularForm()
	{
		String s = "work";
		assertEquals(LGVerbEn.getPastRegularForm(s), "worked");
		
		s = "hope";
		assertEquals(LGVerbEn.getPastRegularForm(s), "hoped");
		
		s = "stay";
		assertEquals(LGVerbEn.getPastRegularForm(s), "stayed");
		
		s = "enjoy";
		assertEquals(LGVerbEn.getPastRegularForm(s), "enjoyed");
		
		s = "fly";
		assertEquals(LGVerbEn.getPastRegularForm(s), "flied");
		
		s = "ski";
		assertEquals(LGVerbEn.getPastRegularForm(s), "skied");
	}
	
	@Test
	public void testGetPastForm() throws Exception
	{
		//LGVerbEn verb = new LGVerbEn(new ZipFile(new File(DTLib.DICTIONARY_JAR)));
		LGVerbEn verb = new LGVerbEn();
		
		assertEquals("foreshowed", verb.getPastForm("foreshow"));
		assertEquals("foreshown", verb.getPastParticipleForm("foreshow"));
		
		assertEquals("hoped", verb.getPastForm("hope"));
		assertEquals("hoped", verb.getPastParticipleForm("hope"));
		
		assertEquals("went", verb.getPastForm("go"));
		assertEquals("gone", verb.getPastParticipleForm("go"));
	}
}
