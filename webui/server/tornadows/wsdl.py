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

""" Class Wsdl to generate WSDL Document """
import xml.dom.minidom
import inspect
from webui.server.tornadows import xmltypes
from webui.server.tornadows import complextypes

class Wsdl:
	""" ToDO:
		- Incorporate exceptions for parameters inputs.
		- When elementInput and/or elementOutput are empty trigger a exception.
	"""
	def __init__(self,nameservice=None,targetNamespace=None,methods=None,location=None):
		self._nameservice = nameservice
		self._namespace = targetNamespace
		self._methods   = methods
		self._location = location

	def createWsdl(self):
		""" Method that allows create the wsdl file """
		typeInput  = None
		typeOutput = None
		types  = '<wsdl:types>\n'
		types += '<xsd:schema targetNamespace="%s">\n'%self._namespace

		namespace = 'xsd'
		types_list = []
		ltype = []
		for wsdl_data in self._methods:
			self._arguments = wsdl_data['args']
			self._elementNameInput = wsdl_data['input'][0]
			self._elementInput = wsdl_data['input'][1]
			self._elementNameOutput = wsdl_data['output'][0]
			self._elementOutput = wsdl_data['output'][1]
			self._operation = wsdl_data['operation']

			method = self._operation

			if len(self._methods) == 1:
				method = ''

			if inspect.isclass(self._elementInput) and issubclass(self._elementInput,complextypes.ComplexType): 
				typeInput = self._elementInput.getName()+method
				
				if ltype.count(self._elementInput.getName()) == 0:
					ltype.append(self._elementInput.getName())
					types += self._elementInput.toXSD(method=method,ltype=ltype)
					
				types += '<%s:element name="%s" type="tns:%s"/>'%(namespace,typeInput,self._elementInput.getName())

			elif isinstance(self._elementInput,dict):
				typeInput = self._elementNameInput+method
				types += self._createComplexTypes(self._elementNameInput+method, self._arguments, self._elementInput)
			elif isinstance(self._elementInput,xmltypes.Array):
				typeInput  = self._elementNameInput+method
				types += self._elementInput.createArray(typeInput)			
			elif isinstance(self._elementInput,list) or inspect.isclass(self._elementInput) and issubclass(self._elementInput,xmltypes.PrimitiveType):
				typeInput  = self._elementNameInput+method
				types += self._createTypes(typeInput,self._elementInput)			
			else: # In case if _elementNameInput is a datatype of python (str, int, float, datetime, etc.) or None
				typeInput  = self._elementNameInput+method
				types += self._createTypes(typeInput,self._elementInput)

			if inspect.isclass(self._elementOutput) and issubclass(self._elementOutput,complextypes.ComplexType): 
				typeOutput = self._elementOutput.getName()+method

				if ltype.count(self._elementOutput.getName()) == 0:
					ltype.append(self._elementOutput.getName())
					types += self._elementOutput.toXSD(method=method,ltype=ltype)

				types += '<%s:element name="%s" type="tns:%s"/>'%(namespace,typeOutput,self._elementOutput.getName())

			elif isinstance(self._elementOutput,xmltypes.Array):
				typeOutput = self._elementNameOutput+method
				types += self._elementOutput.createArray(typeOutput)
			elif isinstance(self._elementOutput,list) or inspect.isclass(self._elementOutput) and issubclass(self._elementOutput,xmltypes.PrimitiveType):
				typeOutput = self._elementNameOutput+method
				types += self._createTypes(typeOutput,self._elementOutput)
			else: # In case if _elementNameOutput is a datatype of python (str, int, float, datetime, etc.) or None
				typeOutput = self._elementNameOutput+method
				types += self._createTypes(typeOutput,self._elementOutput)

			types_list.append({'typeInput':typeInput,'typeOutput':typeOutput,'method':method})

		types += '</xsd:schema>\n'
		types += '</wsdl:types>\n'
		
		messages = ''
		
		for t in types_list:
			typeInput = t['typeInput']
			typeOutput = t['typeOutput']
			method = t['method']

			if len(types_list) == 1:
				method = ''

			messages += '<wsdl:message name="%sRequest%s">\n'%(self._nameservice,method)
			messages += '<wsdl:part name="parameters%s" element="tns:%s"/>\n'%(method,typeInput)
			messages += '</wsdl:message>\n'

			messages += '<wsdl:message name="%sResponse%s">\n'%(self._nameservice,method)
			messages += '<wsdl:part name="returns%s" element="tns:%s"/>\n'%(method,typeOutput)
			messages += '</wsdl:message>\n'

		portType  = '<wsdl:portType name="%sPortType">\n'%self._nameservice
		
		for wsdl_data in self._methods:
			self._operation = wsdl_data['operation']
			
			method = self._operation
			if len(self._methods) == 1:
				method = ''

			portType += '<wsdl:operation name="%s">\n'%self._operation
			portType += '<wsdl:input message="tns:%sRequest%s"/>\n'%(self._nameservice,method)
			portType += '<wsdl:output message="tns:%sResponse%s"/>\n'%(self._nameservice,method)
			portType += '</wsdl:operation>\n'
		
		portType += '</wsdl:portType>\n'

		binding  = '<wsdl:binding name="%sBinding" type="tns:%sPortType">\n'%(self._nameservice,self._nameservice)
		binding += '<soap:binding style="document" transport="http://schemas.xmlsoap.org/soap/http"/>\n'

		for wsdl_data in self._methods:
			self._operation = wsdl_data['operation']

			binding += '<wsdl:operation name="%s">\n'%self._operation		
			binding += '<soap:operation soapAction="%s/%s" style="document"/>\n'%(self._location,self._operation)
			binding += '<wsdl:input><soap:body use="literal"/></wsdl:input>\n'
			binding += '<wsdl:output><soap:body use="literal"/></wsdl:output>\n'
			binding += '</wsdl:operation>\n'

		binding += '</wsdl:binding>\n'
		
		service  = '<wsdl:service name="%s">\n'%self._nameservice
		service += '<wsdl:port name="%sPort" binding="tns:%sBinding">\n'%(self._nameservice,self._nameservice)
		service += '<soap:address location="%s"/>\n'%self._location
		service += '</wsdl:port>\n'
		service += '</wsdl:service>\n'

		definitions  = '<wsdl:definitions name="%s"\n'%self._nameservice
		definitions  += 'xmlns:xsd="http://www.w3.org/2001/XMLSchema"\n'
		definitions  += 'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n'
		definitions  += 'xmlns:tns="%s"\n'%self._namespace
		definitions  += 'xmlns:soap="http://schemas.xmlsoap.org/wsdl/soap/"\n'
		definitions  += 'xmlns:wsdl="http://schemas.xmlsoap.org/wsdl/"\n'
		definitions  += 'targetNamespace="%s">\n'%self._namespace
		definitions += types
		definitions += messages
		definitions += portType
		definitions += binding
		definitions += service
		definitions += '</wsdl:definitions>\n'
		wsdlXml = xml.dom.minidom.parseString(definitions)

		return wsdlXml

	def _createTypes(self, name, elements):
		""" Private method that creates the types for the elements of wsdl """
		elem = ''
		if isinstance(elements,list):
			elem = '<xsd:complexType name="%sParams">\n'%name
			elem += '<xsd:sequence>\n'
			elems = ''
			idx = 1
			for e in elements:
				if hasattr(e,'__name__'):
					elems += '<xsd:element name="value%d" type="xsd:%s"/>\n'%(idx,complextypes.createPythonType2XMLType(e.__name__))
				else:
					elems += e.createElement('value%s'%idx)+'\n'
				idx += 1
			elem += elems+'</xsd:sequence>\n'
			elem += '</xsd:complexType>\n'
			elem += '<xsd:element name="%s" type="tns:%sParams"/>\n'%(name,name)
		elif inspect.isclass(elements) and issubclass(elements,xmltypes.PrimitiveType):
			elem = elements.createElement(name)+'\n'
		elif hasattr(elements,'__name__'):
			elem += '<xsd:element name="%s" type="xsd:%s"/>\n'%(name,complextypes.createPythonType2XMLType(elements.__name__))

		return elem

	def _createComplexTypes(self, name, arguments, elements):
		""" Private method that creates complex types for wsdl """
		elem = ''
		if isinstance(elements,dict):
			elem = '<xsd:complexType name="%sTypes">\n'%name
			elem += '<xsd:sequence>\n'
			elems = ''
			for e in arguments:
				if  isinstance(elements[e],xmltypes.Array):
					elems += elements[e].createType(e)
				elif issubclass(elements[e],xmltypes.PrimitiveType):
					elems += elements[e].createElement(e)+'\n'
				else:
					elems += '<xsd:element name="%s" type="xsd:%s"/>\n'%(e,complextypes.createPythonType2XMLType(elements[e].__name__))
			elem += elems+'</xsd:sequence>\n'
			elem += '</xsd:complexType>\n'
			elem += '<xsd:element name="%s" type="tns:%sTypes"/>\n'%(name,name)
		elif issubclass(elements,xmltypes.PrimitiveType):
			elem = elements.createElement(name)+'\n'

		return elem
