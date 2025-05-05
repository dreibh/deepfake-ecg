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
# Generator Library
# Copyright (C) 2021-2025 by Vajira Thambawita
# Copyright (C) 2021-2025 by Turtle <erencemayez@gmail.com>
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
# * Vajira Thambawita <vlbthambawita@gmail.com>
# * Turtle <erencemayez@gmail.com>
# * Thomas Dreibholz <dreibh@simula.no>

import numpy
import os
import pathlib
import sys
import torch
import tqdm
import typing

from . import Generator


# ------ Constants ----------------------------------------
ECG_SAMPLING_RATE = 500   # in Hz

# ------ ECG types ----------------------------------------
DATA_ECG8         = 8
DATA_ECG12        = 12

# ------ Output formats -----------------------------------
OUTPUT_NUMPY      = 1
OUTPUT_ASC        = 2
OUTPUT_CSV        = 3


# ###### Generate Deepfake ECGs #############################################
def generateDeepfakeECGs(numberOfECGs:       int = 1,
                         ecgType:            int = DATA_ECG8,
                         ecgLengthInSeconds: int = 10,
                         outputFormat:       int = OUTPUT_NUMPY,
                         outputFilePattern:  typing.Union[str, pathlib.Path] = None,
                         outputStartID:      int = 0,
                         runOnDevice:        typing.Literal["cpu", "cuda"] = "cuda" if torch.cuda.is_available() else "cpu"):
   """Generate ECG waveforms using deepfakeecg model, with configurable
      data type (8-lead or 12-lead ECG) and output type (numpy, file).

   Args:
      numberOfECGs (int): The number of ECGs to generate
      ecgLengthInSeconds (int): The ECG length in seconds
      outputFormat (int): The format of the output
         OUTPUT_NUMPY: list of NumPy objects
         OUTPUT_ASC: text, as in the original code
         OUTPUT_CSV: CSV, with additional column for time stamp in milliseconds
      outputFilePattern: Pattern for naming output files, with format() placeholder "number", e.g. 'ecg-{number:06d}.csv'
      outputStartID: Start ID for file numbering
      runOnDevice (str): Device to run generation on ("cpu" or "cuda")

   Returns:
      In case of outputFormat OUTPUT_NUMPY:
         List of arrays of shape (ecgLength, n) containing the ECG data.
         For ECG type DATA_ECG8:
            numpy.ndarray: [I, II, V1, V2, V3, V4, V5, V6]
         For ECG type DATA_ECG12:
            numpy.ndarray: [I, II, V1, V2, V3, V4, V5, V6, III, aVL, aVR, aVF]
   """

   # ====== Initialise generator ============================================
   root_dir = pathlib.Path(__file__).parent
   device = torch.device(runOnDevice)

   generator = Generator()
   checkpoint = torch.load(
      os.path.join(root_dir, "checkpoints/g_stat.pt"),
      map_location = device,
      weights_only = True
   )
   generator.load_state_dict(checkpoint["stat_dict"])
   generator.to(device)
   generator.eval()

   # ====== Make milliseconds time stamp tensor =============================
   ecgLengthInSamples = ecgLengthInSeconds * ECG_SAMPLING_RATE
   timeStamp = torch.arange(0, ECG_SAMPLING_RATE * ecgLengthInSamples,
                            ECG_SAMPLING_RATE,
                            dtype = torch.int32, device = device)
   # Timestamp shape is [ ecgLengthInSamples ]
   timeStamp = torch.t(timeStamp.reshape(1, ecgLengthInSamples))
   # Now, shape is [ ecgLengthInSamples, 1 ]

   # ====== Generate ECGs ===================================================
   results = [ ]
   for i in tqdm.tqdm(range(outputStartID, outputStartID + numberOfECGs)):
      # ------ Create random noise  -----------------------------------------
      noise = torch.Tensor(1, 8, ecgLengthInSamples, device = device).uniform_(-1, 1)

      # ------ Generate ECG -------------------------------------------------
      generatedECG = generator(noise)
      # Output shape is [1, 8, ecgLengthInSamples].

      # ------ Rescale and convert to integer -------------------------------
      generatedECG = generatedECG * 6000
      generatedECG = generatedECG.int()
      generatedECG = torch.transpose(generatedECG.squeeze(), 0, 1)
      # Now, shape is [ecgLengthInSamples, 8].

      # ------ EGC12 computations -------------------------------------------
      if ecgType == DATA_ECG12:
         # Details and formulae:
         # https://ecgwaves.com/topic/ekg-ecg-leads-electrodes-systems-limb-chest-precordial/

         leadI   = generatedECG[:,0]
         leadII  = generatedECG[:,1]

         # Computations:
         # Lead III = Lead II - Lead I
         # aVL      = (Lead I - Lead III) / 2
         # aVRL     = -(Lead I + Lead II) / 2
         # aVF      = (Lead II + Lead III) / 2
         leadIII = leadII - leadI
         aVL     = (leadI - leadIII) / 2
         aVRL    = -(leadI + leadII) / 2
         aVF     = (leadII + leadIII) / 2
         # Shape is [ ecgLengthInSamples ]

         # Reshape to [ ecgLengthInSamples, 1 ] and combine with generatedECG:
         generatedECG = torch.cat( ( generatedECG,
                                     leadIII.reshape(ecgLengthInSamples, 1),
                                     aVL.reshape(ecgLengthInSamples, 1),
                                     aVRL.reshape(ecgLengthInSamples, 1),
                                     aVF.reshape(ecgLengthInSamples, 1)
                                   ) , 1 )

      # ------ Add time stamp for CSV output --------------------------------
      if outputFormat == OUTPUT_CSV:
         # Combine time stamp with generated ECG samples.
         # Now, shape is [ecgLengthInSamples, 1+8].
         generatedECG = torch.cat( (timeStamp, generatedECG), 1 )
         # print(generatedECG[:,0])

      # ------ Make NumPy data ----------------------------------------------
      data = generatedECG.detach().cpu().numpy()

      # ------ Write output file --------------------------------------------
      if outputFormat in [ OUTPUT_ASC, OUTPUT_CSV ]:
        outputFileName = outputFilePattern.format(number = i)
        if outputFormat == OUTPUT_ASC:
           numpy.savetxt(outputFileName, data, fmt = '%i')
        elif outputFormat == OUTPUT_CSV:
           if ecgType == DATA_ECG8:
              header = 'Timestamp,LeadI,LeadII,V1,V2,V3,V4,V5,V6'
           elif ecgType == DATA_ECG12:
              header = 'Timestamp,LeadI,LeadII,V1,V2,V3,V4,V5,V6,LeadIII,aVL,aVR,aVF'
           else:
              raise Exception('Invalid ECG type!')
           numpy.savetxt(outputFileName, data,
                         header    = header,
                         comments  = '',
                         delimiter = ',',
                         fmt       = '%i')

      # ------ Collect data in array ----------------------------------------
      elif outputFormat == OUTPUT_NUMPY:
         results.append(data)
      else:
         raise Exception('Invalid output format!')

   return results


# ###### Generate Deepfake ECG as files #####################################
def generate(num_of_sample: int,
             out_dir:       typing.Union[str, pathlib.Path],
             start_id:      int = 0,
             runOnDevice:   typing.Literal["cpu", "cuda"] = "cuda" if torch.cuda.is_available() else "cpu") -> None:
   """Generate multiple 8-lead ECG waveforms and save them as ASCII files

   Args:
      num_of_sample (int): Number of ECG samples to generate
      out_dir (typing.Union[str, pathlib.Path]): Output directory path where files will be saved
      start_id (int): Starting ID for the generated samples
      runOnDevice (typing.Literal["cpu", "cuda"]): Device to run generation on ("cpu" or "cuda")

   Returns:
      None: Files are saved to the specified output directory with names {start_id}.asc to {start_id + num_of_sample - 1}.asc
      Each file contains ECG data in ASCII format with shape (5000, 8) for leads [I, II, V1, V2, V3, V4, V5, V6]
    """

   generateDeepfakeECGs(num_of_sample, DATA_ECG8,
                        int(5000 / ECG_SAMPLING_RATE), OUTPUT_ASC,
                        os.path.join(out_dir, "{number}.asc"), 0)


# ###### Generate Deepfake ECG as NumPy object ##############################
def generate_as_numpy(runOnDevice: typing.Literal["cpu", "cuda"] = "cuda" if torch.cuda.is_available() else "cpu") -> numpy.ndarray:
   """Generate a single 8-lead ECG waveform using deepfakeecg model

   Args:
       runOnDevice (str): Device to run generation on ("cpu" or "cuda")

   Returns:
       numpy.ndarray: Array of shape (5000, 8) containing the ECG data for leads [I, II, V1, V2, V3, V4, V5, V6]
    """

   results = generateDeepfakeECGs(1, DATA_ECG8,
                                  int(5000 / ECG_SAMPLING_RATE),
                                  OUTPUT_NUMPY)
   return results[0]
