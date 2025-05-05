import numpy
import os
import pathlib
import sys
import torch
import tqdm
from typing  import Union, Literal

from . import Generator


# ###### Generate Deepfake ECG as files #####################################
def generate(num_of_sample: int,
             out_dir:       Union[str, pathlib.Path],
             start_id:      int = 0,
             runOnDevice:   Literal["cpu", "cuda"] = "cuda" if torch.cuda.is_available() else "cpu") -> None:
   """Generate multiple 8-lead ECG waveforms and save them as ASCII files

   Args:
      num_of_sample (int): Number of ECG samples to generate
      out_dir (Union[str, pathlib.Path]): Output directory path where files will be saved
      start_id (int): Starting ID for the generated samples
      runOnDevice (Literal["cpu", "cuda"]): Device to run generation on ("cpu" or "cuda")

   Returns:
      None: Files are saved to the specified output directory with names {start_id}.asc to {start_id + num_of_sample - 1}.asc
      Each file contains ECG data in ASCII format with shape (5000, 8) for leads [I, II, V1, V2, V3, V4, V5, V6]
    """

   generateDeepfakeECGs(num_of_sample, DATA_ECG8, 5000, OUTPUT_ASC,
                        os.path.join(out_dir, "{number}.asc"), 0)


# ###### Generate Deepfake ECG as NumPy object ##############################
def generate_as_numpy(runOnDevice: Literal["cpu", "cuda"] = "cuda" if torch.cuda.is_available() else "cpu") -> numpy.ndarray:
   """Generate a single 8-lead ECG waveform using deepfakeecg model

   Args:
       runOnDevice (str): Device to run generation on ("cpu" or "cuda")

   Returns:
       numpy.ndarray: Array of shape (5000, 8) containing the ECG data for leads [I, II, V1, V2, V3, V4, V5, V6]
    """

   results = generateDeepfakeECGs(1, DATA_ECG8, 5000, OUTPUT_NUMPY)
   return results[0]


# ------ Constants ----------------------------------------
ECG_SAMPLING_RATE = 500

# ------ ECG types ----------------------------------------
DATA_ECG8         = 8
DATA_ECG12        = 12

# ------ Output formats -----------------------------------
OUTPUT_NUMPY      = 1
OUTPUT_ASC        = 2
OUTPUT_CSV        = 3


# ###### Generate Deepfake ECGs #############################################
def generateDeepfakeECGs(numberOfECGs:      int = 1,
                         ecgType:           int = DATA_ECG8,
                         ecgLength:         int = 5000,
                         outputFormat:      int = OUTPUT_NUMPY,
                         outputFilePattern: Union[str, pathlib.Path] = None,
                         outputStartID:     int = 0,
                         runOnDevice:       Literal["cpu", "cuda"] = "cuda" if torch.cuda.is_available() else "cpu"):
   """Generate ECG waveforms using deepfakeecg model, with configurable
      data type (8-lead or 12-lead ECG) and output type (numpy, file).

   Args:
      runOnDevice (str): Device to run generation on ("cpu" or "cuda")

   Returns:
      numpy.ndarray: Array of shape (ecgLength, 8) containing the ECG data
                    for leads [I, II, V1, V2, V3, V4, V5, V6]
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
   timeStamp = torch.arange(0, ECG_SAMPLING_RATE * ecgLength,
                            ECG_SAMPLING_RATE,
                            dtype = torch.int32, device = device)
   # Timestamp shape is [ ecgLength ]
   timeStamp = torch.t(timeStamp.reshape(1, ecgLength))
   # Now, shape is [ 1, ecgLength ]

   # ====== Generate ECGs ===================================================
   results = [ ]
   for i in tqdm.tqdm(range(outputStartID, outputStartID + numberOfECGs)):
      # ------ Create random noise  -----------------------------------------
      noise = torch.Tensor(1, 8, ecgLength, device = device).uniform_(-1, 1)

      # ------ Generate ECG -------------------------------------------------
      generatedECG = generator(noise)
      # Output shape is [1, 8, ecgLength].

      # ------ Rescale and convert to integer -------------------------------
      generatedECG = generatedECG * 6000
      generatedECG = generatedECG.int()
      generatedECG = torch.transpose(generatedECG.squeeze(), 0, 1)
      # Now, shape is [ecgLength, 8].

      # ------ Add time stamp for CSV output --------------------------------
      if outputFormat == OUTPUT_CSV:
         # Combine time stamp with generated ECG samples.
         # Now, shape is [ecgLength, 1+8].
         generatedECG = torch.cat( (timeStamp, generatedECG), 1 )

      # ------ EGC12 computations -------------------------------------------
      if ecgType == DATA_ECG12:
         # Details and formulae:
         # https://ecgwaves.com/topic/ekg-ecg-leads-electrodes-systems-limb-chest-precordial/

         # Lead III = Lead II - Lead I
         # aVL      = (Leaf I - Lead III) / 2
         # aVRL     = -(Leaf I + Lead II) / 2
         # aVF      = (Lead II + Lead III) / 2

         TBD

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
                         header    = 'Timestamp,LeadI,LeadII,V1,V2,V3,V4,V5,V6',
                         comments  = '',
                         delimiter = ',',
                         fmt       = '%i')

      # ------ Collect data in array ----------------------------------------
      elif outputFormat == OUTPUT_NUMPY:
         results.append(data)
      else:
         raise Exception('Invalid output format!')

   return results
