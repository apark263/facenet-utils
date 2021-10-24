#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup, __version__
from pkg_resources import parse_version

minimum_version = parse_version('42.0.0')

if parse_version(__version__) < minimum_version:
    raise RuntimeError(
        f'Package setuptools must be at least version {minimum_version}')

setup()
