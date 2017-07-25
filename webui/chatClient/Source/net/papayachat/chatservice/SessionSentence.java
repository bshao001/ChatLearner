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
import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlElement;
import javax.xml.bind.annotation.XmlType;

/**
 * <p>Java class for SessionSentence complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType name="SessionSentence">
 *   &lt;complexContent>
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType">
 *       &lt;sequence>
 *         &lt;element name="sentence" type="{http://www.w3.org/2001/XMLSchema}string"/>
 *         &lt;element name="sessionId" type="{http://www.w3.org/2001/XMLSchema}integer"/>
 *       &lt;/sequence>
 *     &lt;/restriction>
 *   &lt;/complexContent>
 * &lt;/complexType>
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "SessionSentence", propOrder = {
    "sentence",
    "sessionId"
})
public class SessionSentence {

    @XmlElement(required = true)
    protected String sentence;
    @XmlElement(required = true)
    protected BigInteger sessionId;

    /**
     * Gets the value of the sentence property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getSentence() {
        return sentence;
    }

    /**
     * Sets the value of the sentence property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setSentence(String value) {
        this.sentence = value;
    }

    /**
     * Gets the value of the sessionId property.
     * 
     * @return
     *     possible object is
     *     {@link BigInteger }
     *     
     */
    public BigInteger getSessionId() {
        return sessionId;
    }

    /**
     * Sets the value of the sessionId property.
     * 
     * @param value
     *     allowed object is
     *     {@link BigInteger }
     *     
     */
    public void setSessionId(BigInteger value) {
        this.sessionId = value;
    }
}