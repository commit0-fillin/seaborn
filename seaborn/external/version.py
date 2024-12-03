"""Extract reference documentation from the pypa/packaging source tree.

In the process of copying, some unused methods / classes were removed.
These include:

- parse()
- anything involving LegacyVersion

This software is made available under the terms of *either* of the licenses
found in LICENSE.APACHE or LICENSE.BSD. Contributions to this software is made
under the terms of *both* these licenses.

Vendored from:
- https://github.com/pypa/packaging/
- commit ba07d8287b4554754ac7178d177033ea3f75d489 (09/09/2021)
"""
import collections
import itertools
import re
from typing import Callable, Optional, SupportsInt, Tuple, Union
__all__ = ['Version', 'InvalidVersion', 'VERSION_PATTERN']

class InfinityType:

    def __repr__(self) -> str:
        return 'Infinity'

    def __hash__(self) -> int:
        return hash(repr(self))

    def __lt__(self, other: object) -> bool:
        return False

    def __le__(self, other: object) -> bool:
        return False

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def __ne__(self, other: object) -> bool:
        return not isinstance(other, self.__class__)

    def __gt__(self, other: object) -> bool:
        return True

    def __ge__(self, other: object) -> bool:
        return True

    def __neg__(self: object) -> 'NegativeInfinityType':
        return NegativeInfinity
Infinity = InfinityType()

class NegativeInfinityType:

    def __repr__(self) -> str:
        return '-Infinity'

    def __hash__(self) -> int:
        return hash(repr(self))

    def __lt__(self, other: object) -> bool:
        return True

    def __le__(self, other: object) -> bool:
        return True

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def __ne__(self, other: object) -> bool:
        return not isinstance(other, self.__class__)

    def __gt__(self, other: object) -> bool:
        return False

    def __ge__(self, other: object) -> bool:
        return False

    def __neg__(self: object) -> InfinityType:
        return Infinity
NegativeInfinity = NegativeInfinityType()
InfiniteTypes = Union[InfinityType, NegativeInfinityType]
PrePostDevType = Union[InfiniteTypes, Tuple[str, int]]
SubLocalType = Union[InfiniteTypes, int, str]
LocalType = Union[NegativeInfinityType, Tuple[Union[SubLocalType, Tuple[SubLocalType, str], Tuple[NegativeInfinityType, SubLocalType]], ...]]
CmpKey = Tuple[int, Tuple[int, ...], PrePostDevType, PrePostDevType, PrePostDevType, LocalType]
LegacyCmpKey = Tuple[int, Tuple[str, ...]]
VersionComparisonMethod = Callable[[Union[CmpKey, LegacyCmpKey], Union[CmpKey, LegacyCmpKey]], bool]
_Version = collections.namedtuple('_Version', ['epoch', 'release', 'dev', 'pre', 'post', 'local'])

class InvalidVersion(ValueError):
    """
    An invalid version was found, users should refer to PEP 440.
    """

class _BaseVersion:
    _key: Union[CmpKey, LegacyCmpKey]

    def __hash__(self) -> int:
        return hash(self._key)

    def __lt__(self, other: '_BaseVersion') -> bool:
        if not isinstance(other, _BaseVersion):
            return NotImplemented
        return self._key < other._key

    def __le__(self, other: '_BaseVersion') -> bool:
        if not isinstance(other, _BaseVersion):
            return NotImplemented
        return self._key <= other._key

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _BaseVersion):
            return NotImplemented
        return self._key == other._key

    def __ge__(self, other: '_BaseVersion') -> bool:
        if not isinstance(other, _BaseVersion):
            return NotImplemented
        return self._key >= other._key

    def __gt__(self, other: '_BaseVersion') -> bool:
        if not isinstance(other, _BaseVersion):
            return NotImplemented
        return self._key > other._key

    def __ne__(self, other: object) -> bool:
        if not isinstance(other, _BaseVersion):
            return NotImplemented
        return self._key != other._key
VERSION_PATTERN = '\n    v?\n    (?:\n        (?:(?P<epoch>[0-9]+)!)?                           # epoch\n        (?P<release>[0-9]+(?:\\.[0-9]+)*)                  # release segment\n        (?P<pre>                                          # pre-release\n            [-_\\.]?\n            (?P<pre_l>(a|b|c|rc|alpha|beta|pre|preview))\n            [-_\\.]?\n            (?P<pre_n>[0-9]+)?\n        )?\n        (?P<post>                                         # post release\n            (?:-(?P<post_n1>[0-9]+))\n            |\n            (?:\n                [-_\\.]?\n                (?P<post_l>post|rev|r)\n                [-_\\.]?\n                (?P<post_n2>[0-9]+)?\n            )\n        )?\n        (?P<dev>                                          # dev release\n            [-_\\.]?\n            (?P<dev_l>dev)\n            [-_\\.]?\n            (?P<dev_n>[0-9]+)?\n        )?\n    )\n    (?:\\+(?P<local>[a-z0-9]+(?:[-_\\.][a-z0-9]+)*))?       # local version\n'

class Version(_BaseVersion):
    _regex = re.compile('^\\s*' + VERSION_PATTERN + '\\s*$', re.VERBOSE | re.IGNORECASE)

    def __init__(self, version: str) -> None:
        match = self._regex.search(version)
        if not match:
            raise InvalidVersion(f"Invalid version: '{version}'")
        self._version = _Version(epoch=int(match.group('epoch')) if match.group('epoch') else 0, release=tuple((int(i) for i in match.group('release').split('.'))), pre=_parse_letter_version(match.group('pre_l'), match.group('pre_n')), post=_parse_letter_version(match.group('post_l'), match.group('post_n1') or match.group('post_n2')), dev=_parse_letter_version(match.group('dev_l'), match.group('dev_n')), local=_parse_local_version(match.group('local')))
        self._key = _cmpkey(self._version.epoch, self._version.release, self._version.pre, self._version.post, self._version.dev, self._version.local)

    def __repr__(self) -> str:
        return f"<Version('{self}')>"

    def __str__(self) -> str:
        parts = []
        if self.epoch != 0:
            parts.append(f'{self.epoch}!')
        parts.append('.'.join((str(x) for x in self.release)))
        if self.pre is not None:
            parts.append(''.join((str(x) for x in self.pre)))
        if self.post is not None:
            parts.append(f'.post{self.post}')
        if self.dev is not None:
            parts.append(f'.dev{self.dev}')
        if self.local is not None:
            parts.append(f'+{self.local}')
        return ''.join(parts)
_local_version_separators = re.compile('[\\._-]')

def _parse_local_version(local: str) -> Optional[LocalType]:
    """
    Takes a string like abc.1.twelve and turns it into ("abc", 1, "twelve").
    """
    if local is None:
        return None
    
    components = []
    for part in _local_version_separators.split(local):
        if not part:
            continue
        try:
            components.append(int(part))
        except ValueError:
            components.append(part)
    
    if not components:
        return None
    
    return tuple(components)
