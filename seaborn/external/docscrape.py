"""Extract reference documentation from the NumPy source tree.

Copyright (C) 2008 Stefan van der Walt <stefan@mentat.za.net>, Pauli Virtanen <pav@iki.fi>

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

 1. Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
 2. Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in
    the documentation and/or other materials provided with the
    distribution.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

"""
import inspect
import textwrap
import re
import pydoc
from warnings import warn
from collections import namedtuple
from collections.abc import Callable, Mapping
import copy
import sys

def strip_blank_lines(l):
    """Remove leading and trailing blank lines from a list of lines"""
    start = 0
    end = len(l)
    
    while start < end and not l[start].strip():
        start += 1
    
    while end > start and not l[end - 1].strip():
        end -= 1
    
    return l[start:end]

class Reader:
    """A line-based string reader.

    """

    def __init__(self, data):
        """
        Parameters
        ----------
        data : str
           String with lines separated by '
'.

        """
        if isinstance(data, list):
            self._str = data
        else:
            self._str = data.split('\n')
        self.reset()

    def __getitem__(self, n):
        return self._str[n]

class ParseError(Exception):

    def __str__(self):
        message = self.args[0]
        if hasattr(self, 'docstring'):
            message = f'{message} in {self.docstring!r}'
        return message
Parameter = namedtuple('Parameter', ['name', 'type', 'desc'])

class NumpyDocString(Mapping):
    """Parses a numpydoc string to an abstract representation

    Instances define a mapping from section title to structured data.

    """
    sections = {'Signature': '', 'Summary': [''], 'Extended Summary': [], 'Parameters': [], 'Returns': [], 'Yields': [], 'Receives': [], 'Raises': [], 'Warns': [], 'Other Parameters': [], 'Attributes': [], 'Methods': [], 'See Also': [], 'Notes': [], 'Warnings': [], 'References': '', 'Examples': '', 'index': {}}

    def __init__(self, docstring, config={}):
        orig_docstring = docstring
        docstring = textwrap.dedent(docstring).split('\n')
        self._doc = Reader(docstring)
        self._parsed_data = copy.deepcopy(self.sections)
        try:
            self._parse()
        except ParseError as e:
            e.docstring = orig_docstring
            raise

    def __getitem__(self, key):
        return self._parsed_data[key]

    def __setitem__(self, key, val):
        if key not in self._parsed_data:
            self._error_location(f'Unknown section {key}', error=False)
        else:
            self._parsed_data[key] = val

    def __iter__(self):
        return iter(self._parsed_data)

    def __len__(self):
        return len(self._parsed_data)
    _role = ':(?P<role>\\w+):'
    _funcbacktick = '`(?P<name>(?:~\\w+\\.)?[a-zA-Z0-9_\\.-]+)`'
    _funcplain = '(?P<name2>[a-zA-Z0-9_\\.-]+)'
    _funcname = '(' + _role + _funcbacktick + '|' + _funcplain + ')'
    _funcnamenext = _funcname.replace('role', 'rolenext')
    _funcnamenext = _funcnamenext.replace('name', 'namenext')
    _description = '(?P<description>\\s*:(\\s+(?P<desc>\\S+.*))?)?\\s*$'
    _func_rgx = re.compile('^\\s*' + _funcname + '\\s*')
    _line_rgx = re.compile('^\\s*' + '(?P<allfuncs>' + _funcname + '(?P<morefuncs>([,]\\s+' + _funcnamenext + ')*)' + ')' + '(?P<trailing>[,\\.])?' + _description)
    empty_description = '..'

    def _parse_see_also(self, content):
        """
        func_name : Descriptive text
            continued text
        another_func_name : Descriptive text
        func_name1, func_name2, :meth:`func_name`, func_name3

        """
        result = []
        current_func = None
        current_desc = []

        for line in content:
            line = line.strip()
            if not line:
                continue

            match = self._func_rgx.match(line)
            if match:
                if current_func:
                    result.append((current_func, ' '.join(current_desc)))
                current_func = match.group('name') or match.group('name2')
                current_desc = []
                rest = line[match.end():].strip()
                if rest and rest[0] == ':':
                    current_desc = [rest[1:].strip()]
            elif current_func:
                current_desc.append(line)

        if current_func:
            result.append((current_func, ' '.join(current_desc)))

        return result

    def _parse_index(self, section, content):
        """
        .. index: default
           :refguide: something, else, and more

        """
        result = {}
        for line in content:
            line = line.strip()
            if line.startswith(':'):
                key, value = line[1:].split(':', 1)
                key = key.strip()
                value = value.strip()
                result[key] = [x.strip() for x in value.split(',')]
        return result

    def _parse_summary(self):
        """Grab signature (if given) and summary"""
        if self._is_at_section():
            return

        # If several signatures present, take the last one
        while True:
            summary = self._doc.read_to_next_empty_line()
            summary_str = " ".join([s.strip() for s in summary]).strip()
            if re.compile('^([\w., ]+=)?\s*[\w\.]+\(.*\)$').search(summary_str):
                self['Signature'] = summary_str
                if not self._is_at_section():
                    continue
            break

        if summary is not None:
            self['Summary'] = summary

        if not self._is_at_section():
            self['Extended Summary'] = self._read_to_next_section()

    def __str__(self, func_role=''):
        out = []
        out += self._str_signature()
        out += self._str_summary()
        out += self._str_extended_summary()
        for param_list in ('Parameters', 'Returns', 'Yields', 'Receives', 'Other Parameters', 'Raises', 'Warns'):
            out += self._str_param_list(param_list)
        out += self._str_section('Warnings')
        out += self._str_see_also(func_role)
        for s in ('Notes', 'References', 'Examples'):
            out += self._str_section(s)
        for param_list in ('Attributes', 'Methods'):
            out += self._str_param_list(param_list)
        out += self._str_index()
        return '\n'.join(out)

def dedent_lines(lines):
    """Deindent a list of lines maximally"""
    if not lines:
        return lines

    # Find the minimum indentation (ignore blank lines)
    min_indent = min(len(line) - len(line.lstrip()) for line in lines if line.strip())

    # Remove the minimum indentation from each line
    return [line[min_indent:] if line.strip() else line for line in lines]

class FunctionDoc(NumpyDocString):

    def __init__(self, func, role='func', doc=None, config={}):
        self._f = func
        self._role = role
        if doc is None:
            if func is None:
                raise ValueError('No function or docstring given')
            doc = inspect.getdoc(func) or ''
        NumpyDocString.__init__(self, doc, config)
        if not self['Signature'] and func is not None:
            func, func_name = self.get_func()
            try:
                try:
                    signature = str(inspect.signature(func))
                except (AttributeError, ValueError):
                    if sys.version_info[0] >= 3:
                        argspec = inspect.getfullargspec(func)
                    else:
                        argspec = inspect.getargspec(func)
                    signature = inspect.formatargspec(*argspec)
                signature = f'{func_name}{signature}'
            except TypeError:
                signature = f'{func_name}()'
            self['Signature'] = signature

    def __str__(self):
        out = ''
        func, func_name = self.get_func()
        roles = {'func': 'function', 'meth': 'method'}
        if self._role:
            if self._role not in roles:
                print(f'Warning: invalid role {self._role}')
            out += f'.. {roles.get(self._role, '')}:: {func_name}\n    \n\n'
        out += super().__str__(func_role=self._role)
        return out

class ClassDoc(NumpyDocString):
    extra_public_methods = ['__call__']

    def __init__(self, cls, doc=None, modulename='', func_doc=FunctionDoc, config={}):
        if not inspect.isclass(cls) and cls is not None:
            raise ValueError(f'Expected a class or None, but got {cls!r}')
        self._cls = cls
        if 'sphinx' in sys.modules:
            from sphinx.ext.autodoc import ALL
        else:
            ALL = object()
        self.show_inherited_members = config.get('show_inherited_class_members', True)
        if modulename and (not modulename.endswith('.')):
            modulename += '.'
        self._mod = modulename
        if doc is None:
            if cls is None:
                raise ValueError('No class or documentation string given')
            doc = pydoc.getdoc(cls)
        NumpyDocString.__init__(self, doc)
        _members = config.get('members', [])
        if _members is ALL:
            _members = None
        _exclude = config.get('exclude-members', [])
        if config.get('show_class_members', True) and _exclude is not ALL:

            def splitlines_x(s):
                if not s:
                    return []
                else:
                    # Split the string into lines and remove any trailing empty lines
                    lines = s.splitlines()
                    while lines and not lines[-1].strip():
                        lines.pop()
                    return lines
            for field, items in [('Methods', self.methods), ('Attributes', self.properties)]:
                if not self[field]:
                    doc_list = []
                    for name in sorted(items):
                        if name in _exclude or (_members and name not in _members):
                            continue
                        try:
                            doc_item = pydoc.getdoc(getattr(self._cls, name))
                            doc_list.append(Parameter(name, '', splitlines_x(doc_item)))
                        except AttributeError:
                            pass
                    self[field] = doc_list
