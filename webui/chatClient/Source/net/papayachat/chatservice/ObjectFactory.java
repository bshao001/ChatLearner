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

import javax.xml.bind.JAXBElement;
import javax.xml.bind.annotation.XmlElementDecl;
import javax.xml.bind.annotation.XmlRegistry;
import javax.xml.namespace.QName;

/**
 * This object contains factory methods for each 
 * Java content interface and Java element interface 
 * generated in the net.papayachat.chatservice package. 
 * <p>An ObjectFactory allows you to programatically 
 * construct new instances of the Java representation 
 * for XML content. The Java representation of XML 
 * content can consist of schema derived interfaces 
 * and classes representing the binding of schema 
 * type definitions, element declarations and model 
 * groups.  Factory methods for each of these are 
 * provided in this class.
 * 
 */
@XmlRegistry
public class ObjectFactory {
    private final static QName _SessionSentence_QNAME = 
			new QName("http://papayachat.net/ChatService", "SessionSentence");
    private final static QName _Params_QNAME = 
			new QName("http://papayachat.net/ChatService", "params");

    /**
     * Create a new ObjectFactory that can be used to create new instances of schema 
	 * derived classes for package: net.papayachat.chatservice
     * 
     */
    public ObjectFactory() {
    }

    /**
     * Create an instance of {@link ParamsTypes }
     * 
     */
    public ParamsTypes createParamsTypes() {
        return new ParamsTypes();
    }

    /**
     * Create an instance of {@link SessionSentence }
     * 
     */
    public SessionSentence createSessionSentence() {
        return new SessionSentence();
    }

    /**
     * Create an instance of {@link JAXBElement }{@code <}{@link SessionSentence }{@code >}}
     * 
     */
    @XmlElementDecl(namespace = "http://papayachat.net/ChatService", name = "SessionSentence")
    public JAXBElement<SessionSentence> createSessionSentence(SessionSentence value) {
        return new JAXBElement<SessionSentence>(_SessionSentence_QNAME, SessionSentence.class, 
				null, value);
    }

    /**
     * Create an instance of {@link JAXBElement }{@code <}{@link ParamsTypes }{@code >}}
     * 
     */
    @XmlElementDecl(namespace = "http://papayachat.net/ChatService", name = "params")
    public JAXBElement<ParamsTypes> createParams(ParamsTypes value) {
        return new JAXBElement<ParamsTypes>(_Params_QNAME, ParamsTypes.class, null, value);
    }
}