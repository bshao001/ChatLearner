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

import java.math.BigInteger;
import java.util.Map;
import javax.xml.ws.BindingProvider;

import com.sun.xml.ws.client.BindingProviderProperties;

public class ChatClient {
	private final ChatServicePortType cPort;
	private int sessionId;
	
	/**
	 * serviceAddress example: http://papayachat.net:8080/ChatService
	 */
	public ChatClient(String serviceAddress, int timeout) {
		ChatService cServ = new ChatService();
		this.cPort = cServ.getChatServicePort();
		this.sessionId = 0;
		
		Map<String, Object> ctxt = ((BindingProvider)cPort).getRequestContext();
		ctxt.put(BindingProvider.ENDPOINT_ADDRESS_PROPERTY, serviceAddress);
		ctxt.put(BindingProviderProperties.CONNECT_TIMEOUT, timeout*1000);
		ctxt.put(BindingProviderProperties.REQUEST_TIMEOUT, timeout*1000);
	}
	
	public String getReply(String question) throws Exception {
		ObjectFactory of = new ObjectFactory();
		ParamsTypes params = of.createParamsTypes();
		params.setSessionId(BigInteger.valueOf(this.sessionId));
		params.setQuestion(question);
		
		SessionSentence resp = cPort.reply(params);
		int newId = resp.getSessionId().intValue();
		if (this.sessionId != newId) setSessionId(newId);
		
		return resp.getSentence();
	}
	
	private void setSessionId(int sessionId) {
		this.sessionId = sessionId;
	}
}