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

""" 
	Are incorporated the primitive datatypes defined by XML.
	Array is defined for the use of array of elements and his respective datatype.
"""

import inspect
from webui.server.tornadows import complextypes

def createElementXML(name,type,prefix='xsd'):
	""" Function used for the creation of xml elements. """
	return '<%s:element name="%s" type="%s:%s"/>'%(prefix,name,prefix,type)

def createArrayXML(name,type,prefix='xsd',maxoccurs=None):
	""" Function used for the creation of xml complexElements """
	complexType  = '<%s:complexType name="%sParams">\n'%(prefix,name)
	complexType += '<%s:sequence>\n'%prefix
	if maxoccurs == None:
		complexType += '<%s:element name="value" type="%s:%s" maxOccurs="unbounded"/>\n'%(prefix,prefix,type)
	else:
		complexType += '<%s:element name="value" type="%s:%s" maxOccurs="%d"/>\n'%(prefix,prefix,type,maxoccurs)
	complexType += '</%s:sequence>\n'%prefix
	complexType += '</%s:complexType>\n'%prefix
	complexType += '<%s:element name="%s" type="tns:%sParams"/>\n'%(prefix,name,name)
	return complexType

class Array:
	""" Create arrays of xml elements.
	    
	    Here an example:

	    @webservices(_params=xmltypes.Array(xmltypes.Integer),_returns=xmltypes.Integer)
	    def function(sefl, list_of_elements):
		for e in list_of_elements:
		# Do something with the element    
        	return len(list_of_elements)

	    xmltypes.Array(xmltype.Integer) generate an xml element into schema definition:
		<xsd:element name="arrayOfElement" type="xsd:integer" maxOccurs="unbounded"/>

	    this make the parameter of the function list_of_elements is a python list.

	    if you specify xmltypes.Array(xmltypes.Integer,10), is generated:
		<xsd:element name="arrayOfElement" type="xsd:integer" maxOccurs="10"/>
	"""
	def __init__(self,type,maxOccurs=None):
		self._type = type
		self._n    = maxOccurs

	def createArray(self,name):
		type = None
		if inspect.isclass(self._type) and not issubclass(self._type,PrimitiveType):
			type = complextypes.createPythonType2XMLType(self._type.__name__)
		else:
			type = self._type.getType(self._type)
		return createArrayXML(name,type,'xsd',self._n)

	def createType(self,name):
		prefix = 'xsd'
		type = None
		if inspect.isclass(self._type) and not issubclass(self._type,PrimitiveType):
			type = complextypes.createPythonType2XMLType(self._type.__name__)
		else:
			type = self._type.getType(self._type)
		maxoccurs = self._n
		complexType = ''
		if self._n == None:
			complexType += '<%s:element name="%s" type="%s:%s" maxOccurs="unbounded"/>\n'%(prefix,name,prefix,type)
		else:
			complexType += '<%s:element name="%s" type="%s:%s" maxOccurs="%d"/>\n'%(prefix,name,prefix,type,maxoccurs)
		return complexType

	def genType(self,v):
		value = None
		if inspect.isclass(self._type) and issubclass(self._type,PrimitiveType):
			value = self._type.genType(v)
		elif hasattr(self._type,'__name__'):
			value = complextypes.convert(self._type.__name__,v)
			# Convert str to bool
			if value == 'true':
				value = True
			elif value == 'false':
				value = False
		return value

class PrimitiveType:
	""" Class father for all derived types. """
	pass

class Integer(PrimitiveType):
	""" 1. XML primitive type : integer """
	@staticmethod
	def createElement(name,prefix='xsd'):
		return createElementXML(name,'integer')
	@staticmethod
	def getType(self):
		return 'integer'
	@classmethod
	def genType(self,v):
		return int(v)

class Decimal(PrimitiveType):
	""" 2. XML primitive type : decimal """
	@staticmethod
	def createElement(name,prefix='xsd'):
		return createElementXML(name,'decimal')
	@staticmethod
	def getType(self):
		return 'decimal'
	@classmethod
	def genType(self,v):
		return float(v)

class Double(PrimitiveType):
	""" 3. XML primitive type : double """
	@staticmethod
	def createElement(name,prefix='xsd'):
		return createElementXML(name,'double')
	@staticmethod
	def getType(self):
		return 'double'
	@classmethod
	def genType(self,v):
		return float(v)

class Float(PrimitiveType):
	""" 4. XML primitive type : float """
	@staticmethod
	def createElement(name,prefix='xsd'):
		return createElementXML(name,'float')
	@staticmethod
	def getType(self):
		return 'float'
	@classmethod
	def genType(self,v):
		return float(v)

class Duration(PrimitiveType):
	""" 5. XML primitive type : duration """
	@staticmethod
	def createElement(name,prefix='xsd'):
		return createElementXML(name,'duration')
	@staticmethod
	def getType(self):
		return 'duration'
	@classmethod
	def genType(self,v):
		return str(v)

class Date(PrimitiveType):
	""" 6. XML primitive type : date """
	@staticmethod
	def createElement(name,prefix='xsd'):
		return createElementXML(name,'date')
	@staticmethod
	def getType(self):
		return 'date'
	@classmethod
	def genType(self,v):
		return str(v)

class Time(PrimitiveType):
	""" 7. XML primitive type : time """
	@staticmethod
	def createElement(name,prefix='xsd'):
		return createElementXML(name,'time')
	@staticmethod
	def getType(self):
		return 'time'
	@classmethod
	def genType(self,v):
		return str(v)

class DateTime(PrimitiveType):
	""" 8. XML primitive type : dateTime """
	@staticmethod
	def createElement(name,prefix='xsd'):
		return createElementXML(name,'dateTime')
	@staticmethod
	def getType(self):
		return 'dateTime'
	@classmethod
	def genType(self,v):
		return str(v)

class String(PrimitiveType):
	""" 9. XML primitive type : string """
	@staticmethod
	def createElement(name,prefix='xsd'):
		return createElementXML(name,'string')
	@staticmethod
	def getType(self):
		return 'string'
	@classmethod
	def genType(self,v):
		return str(v)

class Boolean(PrimitiveType):
	""" 10. XML primitive type : boolean """
	@staticmethod
	def createElement(name,prefix='xsd'):
		return createElementXML(name,'boolean')
	@staticmethod
	def getType(self):
		return 'boolean'
	@classmethod
	def genType(self,v):
		return str(v).lower()
