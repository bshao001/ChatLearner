/*=============================================================================
 * Copyright 2017 Bo Shao. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *	
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
==============================================================================*/
package net.papayachat.chatservice;

import java.net.*;
import java.io.*;

public class ChatClient {
	private final String serviceAddress;
	private final int timeout;
	private int sessionId;
	
	/**
	 * serviceAddress example: http://papayachat.net:5000/reply
	 */
	public ChatClient(String serviceAddress, int timeout) {
		this.serviceAddress = serviceAddress;
		this.timeout = timeout;
		this.sessionId = 0;
	}
	
	public String getReply(String question) throws Exception {
		BufferedReader bReader = null;
	
		try {
			String serviceUrl = serviceAddress + "?sessionId="+sessionId+"&question=" + 
					URLEncoder.encode(question, "UTF-8");
			URL url = new URL(serviceUrl);
			URLConnection uc = url.openConnection();
		
			uc.setConnectTimeout(timeout*1000);
			uc.setReadTimeout(timeout*1000);
		
			bReader = new BufferedReader(new InputStreamReader(uc.getInputStream()));	
           
			String inLine = null;
			StringBuilder resp = new StringBuilder();
			
			while ((inLine = bReader.readLine()) != null) {
				resp.append(inLine).append("\n");
			}
			
			javax.json.JsonReader jr = javax.json.Json.createReader(new StringReader(resp.toString()));
			javax.json.JsonObject jo = jr.readObject();

			int newId = jo.getInt("sessionId");
			if (sessionId != newId) setSessionId(newId);
			
			String answer = jo.getString("sentence");
			
			return answer;
        } catch (Exception e) {
			e.printStackTrace();
			
        	return "Chat service not available at this moment. Please try again later!";
        } finally {
			if (bReader != null) bReader.close(); 
		}
	}
	
	public int getSessionId() {
		return sessionId;
	}
	
	private void setSessionId(int sessionId) {
		this.sessionId = sessionId;
	}
}