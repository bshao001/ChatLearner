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

""" Implementation of module with classes and functions for transform python 
    classes in xml schema: 

    See the next example:

    	from tornadows.complextypes import ComplexType, StringProperty, IntegerProperty

    	class Person(ComplexType):
		name = StringProperty()
		age  = IntegerProperty()

	or you can use some python types

	class Person(ComplexType):
		name = str
		age  = int

	is equivalent to:

	 <xsd:complexType name="Person">
		<xsd:sequence>
			<xsd:element name="name" type="xsd:string"/>
			<xsd:element name="age" type="xsd:integer"/> 
		</xsd:sequence>
	 </xsd:complexType>

"""

import webui.server.tornadows.xmltypes
import xml.dom.minidom
import inspect
from datetime import date, datetime, time
	
class Property:
	""" Class base for definition of properties of the attributes of a python class """
	pass

class IntegerProperty(Property):
	""" Class for definitions of Integer Property """
	def __init__(self):
		self.type = webui.server.tornadows.xmltypes.Integer
		self.value = None

class DecimalProperty(Property):
	""" Class for definitions of Decimal Property """
	def __init__(self):
		self.type = webui.server.tornadows.xmltypes.Decimal
		self.value = None

class DoubleProperty(Property):
	""" Class for definitions of Double Property """
	def __init__(self):
		self.type = webui.server.tornadows.xmltypes.Double
		self.value = None

class FloatProperty(Property):
	""" Class for definitions of Float Property """
	def __init__(self):
		self.type = webui.server.tornadows.xmltypes.Float
		self.value = None

class DurationProperty(Property):
	""" Class for definitions of Duration Property """
	def __init__(self):
		self.type = webui.server.tornadows.xmltypes.Duration
		self.value = None

class DateProperty(Property):
	""" Class for definitions of Date Property """
	def __init__(self):
		self.type = webui.server.tornadows.xmltypes.Date
		self.value = None

class TimeProperty(Property):
	""" Class for definitions of Time Property """
	def __init__(self):
		self.type = webui.server.tornadows.xmltypes.Time
		self.value = None

class DateTimeProperty(Property):
	""" Class for definitions of DateTime Property """
	def __init__(self):
		self.type = webui.server.tornadows.xmltypes.DateTime
		self.value = None

class StringProperty(Property):
	""" Class for definitions of String Property """
	def __init__(self):
		self.type = webui.server.tornadows.xmltypes.String
		self.value = None

class BooleanProperty(Property):
	""" Class for definitions of Boolean Property """
	def __init__(self):
		self.type = webui.server.tornadows.xmltypes.Boolean
		self.value = None

class ArrayProperty(list):
	""" For create a list of classes """
	def __init__(self, object, minOccurs = 1, maxOccurs=None, data=[]):
		list.__init__(self,data)
		self._minOccurs = minOccurs
		self._maxOccurs = maxOccurs
		self._object = object
		self.append(self._object)
		
	def toXSD(self,namespace='xsd',nameelement=None):
		""" Create xml complex type for ArrayProperty """
		xsd = self._object.toXSD()
		if self._maxOccurs == None:
			xsd += '<%s:element name="%s" type="tns:%s" minOccurs="%s"/>'%(namespace,nameelement,self._object.getName(),self._minOccurs)
		elif self._maxOccurs != None:
			xsd += '<%s:element name="%s" type="tns:%s" minOccurs="%s" maxOccurs="%s"/>'%(namespace,nameelement,self._object.getName(),str(self._minOccurs),str(self._maxOccurs))
		return xsd

class ComplexType(object):
	""" Base class for definitions of python class like xml document and schema:

	    from webui.server.tornadows.complextypes import ComplexType,StringProperty, IntegerProperty

	    class Person(ComplexType):
		name = StringProperty
		age  = IntegerProperty
	
	    if __name__ == '__main__':
		print 'XML Schema : '
		print(Person.toXSD())
		
		p = Person()
		p.name.value = 'Steve J'
		p.age.value  = 38

		print('XML Document : ')
		print(p.toXML())

	    or you if you want to use some python types (int, str, float, bool)

	    from webui.server.tornadows.complextypes import ComplexType

	    class Person(ComplexType):
		name = str 
		age  = int
	
	    if __name__ == '__main__':
		print('XML Schema : ')
		print(Person.toXSD())
		
		p = Person()
		p.name.value = 'Steve J'
		p.age.value  = 38

		print('XML Document : ')
		print(p.toXML())

	"""
	def __init__(self):
		""" Class constructor for ComplexType """
		default_attr = dir(type('default',(object,),{}))
		for attr in self.__class__.__dict__.keys():
			if default_attr.count(attr) > 0 or callable(attr):
				continue	
			else:
				element = self.__class__.__dict__[attr]
				typeobj = self._createAttributeType(element)
				setattr(self,attr,typeobj)

	def toXML(self,name=None,method=''):
		""" Method that creates the XML document for the instance of python class.
		    Return a string with the xml document.
		 """
		nameroot = None

		if name == None:
			nameroot = self.__class__.__name__
		else:
			nameroot = name
		nameroot += method

		xml = '<%s>'%nameroot
		default_attr = dir(type('default',(object,),{}))
		for key in dir(self):
			if default_attr.count(key) > 0:
				continue
			element = findElementFromDict(self.__dict__,key)
			if element == None:
				continue
			if isinstance(element,list):
				for e in element:
					if isinstance(e,ComplexType):
						xml += e.toXML(name=key)
					else:
						xml += '<%s>%s</%s>'%(key,e,key)
			elif isinstance(element,Property):
				xml += '<%s>%s</%s>'%(key,element.value,key)
			elif isinstance(element,ComplexType):
				xml += element.toXML(name=key)
			else:
				xml += '<%s>%s</%s>'%(key,convert(type(element).__name__,element),key)
		xml += '</%s>'%nameroot
		return str(xml)
					
	@classmethod
	def toXSD(cls,xmlns='http://www.w3.org/2001/XMLSchema',namespace='xsd',method='', ltype=[]):
		""" Class method that creates the XSD document for the python class.
		    Return a string with the xml schema.
		 """
		name = cls.__name__
		xsd  = cls._generateXSD(ltype=ltype)
		return xsd
		
	@classmethod	
	def _generateXSD(cls,xmlns='http://www.w3.org/2001/XMLSchema',namespace='xsd', ltype=[]):
		""" Class method for get the xml schema with the document definition.
		    Return a string with the xsd document.
		 """
		default_attr = dir(type('default',(object,),{}))
		name = cls.__name__
		xsd  = '<%s:complexType name="%s" xmlns:%s="%s">'%(namespace,name,namespace,xmlns)
		xsd += '<%s:sequence>'%namespace
		complextype = []

		for key in dir(cls):
			if default_attr.count(key) > 0:
				continue
			element = findElementFromDict(cls.__dict__,key)
			if element == None:
				continue
			if isinstance(element,Property):
				xsd += element.type.createElement(str(key))
			
			elif isinstance(element,ComplexType): 
				nameinstance = key

				if ltype.count(self._elementInput.getName()) == 0:
					ltype.append(self._elementInput.getName())
					complextype.append(element._generateXSD(ltype=[]))
				
				xsd += '<%s:element name="%s" type="tns:%s"/>'%(namespace,nameinstance,element.getName())			
			elif inspect.isclass(element) and issubclass(element,ComplexType): 
				nameinstance = key
				
				if ltype.count(element.getName()) == 0:
					ltype.append(element.getName())
					complextype.append(element._generateXSD(ltype=[]))
				
				xsd += '<%s:element name="%s" type="tns:%s"/>'%(namespace,nameinstance,element.getName())			
			elif isinstance(element,ArrayProperty):
				if isinstance(element[0],ComplexType) or issubclass(element[0],ComplexType):
					complextype.append(element[0]._generateXSD(ltype=[]))
					xsd += '<%s:element name="%s" type="tns:%s" maxOccurs="unbounded"/>'%(namespace,key,element[0].__name__)	
				else:
					typeelement = createPythonType2XMLType(element[0].__name__)
					xsd += '<%s:element name="%s" type="%s:%s" maxOccurs="unbounded"/>'%(namespace,key,namespace,typeelement)	
			
			elif isinstance(element,list):
				if isinstance(element[0],ComplexType) or issubclass(element[0],ComplexType):

					if ltype.count(element[0].__name__) == 0:
						ltype.append(element[0].__name__)
						complextype.append(element[0]._generateXSD(ltype=[]))
					
					xsd += '<%s:element name="%s" type="tns:%s" maxOccurs="unbounded"/>'%(namespace,key,element[0].__name__)	
				else:
					typeelement = createPythonType2XMLType(element[0].__name__)
					xsd += '<%s:element name="%s" type="%s:%s" maxOccurs="unbounded"/>'%(namespace,key,namespace,typeelement)	
			elif hasattr(element,'__name__'):
				typeelement = createPythonType2XMLType(element.__name__)
				xsd += '<%s:element name="%s" type="%s:%s"/>'%(namespace,str(key),namespace,typeelement)

		xsd += '</%s:sequence>'%namespace
		xsd += '</%s:complexType>'%namespace
		
		if len(complextype) > 0:
			for ct in complextype:
				xsd += ct
				
		return xsd
		
	@classmethod
	def getName(cls):
		""" Class method return the name of the class """
		return cls.__name__
		
	@classmethod	
	def _createAttributeType(self,element):
		""" Class method to create the types of the attributes of a ComplexType """
		if isinstance(element,list):
			return list()
		elif isinstance(element,IntegerProperty):
			return IntegerProperty()
		elif isinstance(element,DecimalProperty):
			return DecimalProperty()
		elif isinstance(element,DoubleProperty):
			return DoubleProperty()
		elif isinstance(element,FloatProperty):
			return FloatProperty()
		elif isinstance(element,DurationProperty):
			return DurationProperty()
		elif isinstance(element,DateProperty):
			return DateProperty()
		elif isinstance(element,TimeProperty):
			return TimeProperty()
		elif isinstance(element,DateTimeProperty):
			return DateTimeProperty()
		elif isinstance(element,StringProperty):
			return StringProperty()
		elif isinstance(element,BooleanProperty):
			return BooleanProperty()
		elif issubclass(element,ComplexType):
			return element()
		else:
			if   element.__name__ == 'int':	
				return int
			elif element.__name__ == 'decimal':
				return float
			elif element.__name__ == 'double':
				return float
			elif element.__name__ == 'float':
				return float
			elif element.__name__ == 'duration':
				return str
			elif element.__name__ == 'date':
				return date
			elif element.__name__ == 'time':
				return time
			elif element.__name__ == 'dateTime':
				return datetime
			elif element.__name__ == 'str':
				return str
			elif element.__name__ == 'bool':
				return bool

def xml2object(xml,xsd,complex,method=''):
	""" Function that converts a XML document in a instance of a python class """
	namecls = complex.getName()
	types   = xsd2dict(xsd)
	lst     = xml2list(xml,namecls,types,method=method)
	tps     = cls2dict(complex)
	obj     = generateOBJ(lst,namecls,tps)
	return obj

def cls2dict(complex):
	""" Function that creates a dictionary from a ComplexType class with the attributes and types """
	default_attr = dir(type('default',(object,),{}))
	dct = {}
	for attr in dir(complex):
		if default_attr.count(attr) > 0 or callable(attr):
			continue
		else:
			elem = findElementFromDict(complex.__dict__,attr)
			if elem != None:
				dct[attr] = elem
	return dct

def xsd2dict(xsd,namespace='xsd'):
	""" Function that creates a dictionary from a xml schema with the type of element """
	types = ['xsd:integer','xsd:decimal','xsd:double','xsd:float','xsd:duration','xsd:date','xsd:time','xsd:dateTime','xsd:string','xsd:boolean']
	dct = {}

	element = '%s:element'%namespace
	elems = xsd.getElementsByTagName(element)
	for e in elems:
		val = 'complexType'
		typ = str(e.getAttribute('type'))
		lst = e.hasAttribute('maxOccurs')
		if types.count(typ) > 0:
			val = 'element'
		dct[str(e.getAttribute('name'))] = (val,typ,lst)
	return dct

def xml2list(xmldoc,name,types,method=''):
	""" Function that creates a list from xml documento with a tuple element and value """
	name = name+method
	
	x = xml.dom.minidom.parseString(xmldoc)
	c = None
	if x.documentElement.prefix != None:
		c = x.getElementsByTagName(x.documentElement.prefix+':'+name)
	else:
		c = x.getElementsByTagName(name)
	attrs = genattr(c)
	lst = []
	for a in attrs:
		t = types[a.nodeName]
		typ = t[0]
		typxml = t[1]
		isarray = t[2]
		if typ == 'complexType' or typ == 'list':
			l = xml2list(a.toxml(),str(a.nodeName),types)
			lst.append((str(a.nodeName),l,isarray))
		else:
			val = None
			if len(a.childNodes) > 0:
				val = convert(typxml,str(a.childNodes[0].nodeValue))
				# Convert str to bool.
				if val == 'true':
					val = True
				elif val == 'false':
					val = False
			lst.append((str(a.nodeName),val,isarray))
	return lst

def generateOBJ(d,namecls,types):
	""" Function that creates a object from a xml document """
	dct = {}
	lst = []
	for a in d:
		name  = a[0]
		value = a[1]
		isarray = a[2]
		if isinstance(value,list):
			o = generateOBJ(value,name,types)
			if isarray:
				lst.append(o)
				dct[name] = lst
			else:
				dct[name] = o
		else:
			typ = findElementFromDict(types,name)
			if isinstance(typ,Property):
				dct[name] = createProperty(typ,value)
			else:
				dct[name] = value
	return type(namecls,(ComplexType,),dct)
	
def createProperty(typ,value):
	""" Function that creates a Property class instance, with the value """
	ct = None
	if isinstance(typ,IntegerProperty):
		ct = IntegerProperty()
		ct.value = webui.server.tornadows.xmltypes.Integer.genType(value)
	elif isinstance(typ,DecimalProperty):
		ct = DecimalProperty()
		ct.value = webui.server.tornadows.xmltypes.Decimal.genType(value)
	elif isinstance(typ,DoubleProperty):
		ct = DoubleProperty()
		ct.value = webui.server.tornadows.xmltypes.Double.genType(value)
	elif isinstance(typ,FloatProperty):
		ct = FloatProperty()
		ct.value = webui.server.tornadows.xmltypes.Float.genType(value)
	elif isinstance(typ,DurationProperty):
		ct = DurationProperty()
		ct.value = webui.server.tornadows.xmltypes.Duration.genType(value)
	elif isinstance(typ,DateProperty):
		ct = DateProperty()
		ct.value = webui.server.tornadows.xmltypes.Date.genType(value)
	elif isinstance(typ,TimeProperty):
		ct = TimeProperty()
		ct.value = webui.server.tornadows.xmltypes.Time.genType(value)
	elif isinstance(typ,DateTimeProperty):
		ct = DateTimeProperty()
		ct.value = webui.server.tornadows.xmltypes.DateTime.genType(value)
	elif isinstance(typ,StringProperty):
		ct = StringProperty()
		ct.value = webui.server.tornadows.xmltypes.String.genType(value)
	elif isinstance(typ,BooleanProperty):
		ct = BooleanProperty()
		ct.value = webui.server.tornadows.xmltypes.Boolean.genType(value)

	return ct

def genattr(elems):
	""" Function that generates a list with the childnodes of a xml element  """
	d = []
	for e in elems[0].childNodes:
		if e.nodeType == e.ELEMENT_NODE:
			d.append(e)
	return d

def findElementFromDict(dictionary,key):
	""" Function to find a element into a dictionary for the key """
	element = None
	try:
		element = dictionary[key]
		return element
	except KeyError:
		return None

def convert(typeelement,value):
	""" Function that converts a value depending his type """
	if typeelement == 'xsd:integer' or typeelement == 'int':	
		return int(value)
	elif typeelement == 'xsd:decimal':
		return float(value)
	elif typeelement == 'xsd:double':
		return float(value)
	elif typeelement == 'xsd:float' or typeelement == 'float':
		return float(value)
	elif typeelement == 'xsd:duration':
		return str(value)
	elif typeelement == 'xsd:date' or typeelement == 'date':
		sdate = str(value).split('-')
		return date(int(sdate[0]),int(sdate[1]),int(sdate[2]))
	elif typeelement == 'xsd:time' or typeelement == 'time':
		stime = str(value).split(':')
		hour = stime[0]
		min  = stime[1]
		seg  = '00'
		if len(stime) >= 3:
			seg = stime[2].split('.')[0]
		return time(int(hour),int(min),int(seg))
	elif typeelement == 'xsd:dateTime' or typeelement == 'datetime':
		sdatetime = str(value).replace('T','-').replace(' ','-').replace('+','-').split('-')
		year  = sdatetime[0]
		mon   = sdatetime[1]
		day   = sdatetime[2]
		stime = sdatetime[3].split(':')
		hour  = stime[0]
		min   = stime[1]
		seg   = '00'
		if len(stime) >= 3:
			seg = stime[2].split('.')[0]
		return datetime(int(year),int(mon),int(day),int(hour),int(min),int(seg)).isoformat('T')
	elif typeelement == 'xsd:string' or typeelement == 'str' or typeelement == 'unicode':
		return str(value)
	elif typeelement == 'xsd:boolean' or typeelement == 'bool':
		return str(value).lower()

def createPythonType2XMLType(pyType):
	""" Function that creates a xml type from a python type """
	xmlType = None
	if pyType == 'int':
		xmlType = 'integer'
	elif pyType == 'decimal':
		xmlType = 'decimal'
	elif pyType == 'double':
		xmlType = 'float'				
	elif pyType == 'float':
		xmlType = 'float'
	elif pyType == 'duration':
		xmlType = 'duration'
	elif pyType == 'date':
		xmlType = 'date'
	elif pyType == 'time':
		xmlType = 'time'
	elif pyType == 'datetime':
		xmlType = 'dateTime'
	elif pyType == 'str':
		xmlType = 'string'
	elif pyType == 'bool':
		xmlType = 'boolean'
		
	return xmlType

