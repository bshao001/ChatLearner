#!/usr/bin/env python
#
# Copyright 2011 Rodrigo Ancavil del Pino
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

""" Implementation of a envelope soap 1.1 """

import xml.dom.minidom

class SoapMessage:
	""" Implementation of a envelope soap 1.1 with minidom API

		import tornadows.soap
		import xml.dom.minidom

		soapenvelope = tornadows.soap.SoapMessage()
		xmlDoc = xml.dom.minidom.parseString('<Doc>Hello, world!!!</Doc>')
		soapenvelope.setBody(xmlDoc)
		for s in soapenvelope.getBody():
			print s.toxml()

	"""
	def __init__(self):
		self._soap = xml.dom.minidom.Document()
		envurl = 'http://schemas.xmlsoap.org/soap/envelope/'
		self._envelope = self._soap.createElementNS(envurl, 'soapenv:Envelope')
		self._envelope.setAttribute('xmlns:soapenv', envurl)
		self._envelope.setAttribute('xmlns:xsi',
				"http://www.w3.org/2001/XMLSchema-instance")
		self._envelope.setAttribute('xsi:schemaLocation',
				' '.join((envurl, envurl)))
		self._soap.appendChild(self._envelope)
		self._header = self._soap.createElement('soapenv:Header')
		self._body   = self._soap.createElement('soapenv:Body')
		self._envelope.appendChild(self._header)
		self._envelope.appendChild(self._body)

	def getSoap(self):
		""" Return the soap envelope as xml.dom.minidom.Document 
		    getSoap() return a xml.dom.minidom.Document object
		"""
		return self._soap

	def getHeader(self):
		""" Return the child elements of Header element 
		    getHeader() return a list with xml.dom.minidom.Element objects
		"""
		return self._header.childNodes

	def getBody(self):
		""" Return the child elements of Body element 
		    getBody() return a list with xml.dom.minidom.Element objects
		"""
		return self._body.childNodes

	def setHeader(self, header):
		""" Set the child content to Header element
		    setHeader(header), header is a xml.dom.minidom.Document object
		 """
		if isinstance(header,xml.dom.minidom.Document):
			self._header.appendChild(header.documentElement)
		elif isinstance(header,xml.dom.minidom.Element):
			self._header.appendChild(header)

	def setBody(self,body):
		""" Set the child content to Body element 
		    setBody(body), body is a xml.dom.minidom.Document object or
		    a xml.dom.minidom.Element
		"""
		if isinstance(body,xml.dom.minidom.Document):
			self._body.appendChild(body.documentElement)
		elif isinstance(body,xml.dom.minidom.Element):
			self._body.appendChild(body)

	def removeHeader(self):
		""" Remove the last child elements from Header element """
		lastElement = self._header.lastChild
		if lastElement != None:
			self._header.removeChild(lastElement)

	def removeBody(self):
		""" Remove last child elements from Body element """
		lastElement = self._body.lastChild
		if lastElement != None:
			self._body.removeChild(lastElement)
