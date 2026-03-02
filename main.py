#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 28 16:11:35 2026

@author: javfdez
"""
from extfunc import import_galaxy_data
import matplotlib.pyplot as plt
import corner

galaxy_select = [1,2]

cube = {}

for i in galaxy_select:
    cube[i] = import_galaxy_data(i)
