#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==========================================================================
#         ____                   __       _          _____ ____ ____
#        |  _ \  ___  ___ _ __  / _| __ _| | _____  | ____/ ___/ ___|
#        | | | |/ _ \/ _ \ '_ \| |_ / _` | |/ / _ \ |  _|| |  | |  _
#        | |_| |  __/  __/ |_) |  _| (_| |   <  __/ | |__| |__| |_| |
#        |____/ \___|\___| .__/|_|  \__,_|_|\_\___| |_____\____\____|
#                        |_|
#
#                       --- Deepfake ECG Generator ---
#                https://github.com/vlbthambawita/deepfake-ecg
# ==========================================================================
#
# Command-Line Generator Tool
# Copyright (C) 2025 by Turtle <erencemayez@gmail.com>
# Copyright (C) 2025 by Thomas Dreibholz
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Contact:
# * Turtle <erencemayez@gmail.com>
# * Thomas Dreibholz <dreibh@simula.no>

import deepfakeecg

deepfakeecg.generate(5, ".", start_id=0)  # Generate 5 ECGs to the current folder starting from id=0
