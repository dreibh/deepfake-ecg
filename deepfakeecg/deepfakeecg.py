import torch
import os
import numpy
from tqdm import tqdm
from . import Generator
from pathlib import Path
from typing import Union, Literal


def generate(num_of_sample: int, out_dir: Union[str, Path], start_id: int = 0,
             runOnDevice: Literal["cpu", "cuda"] = "cuda" if torch.cuda.is_available() else "cpu") -> None:
    """Generate multiple 8-lead ECG waveforms and save them as ASCII files

    Args:
        num_of_sample (int): Number of ECG samples to generate
        out_dir (Union[str, Path]): Output directory path where files will be saved
        start_id (int): Starting ID for the generated samples
        runOnDevice (Literal["cpu", "cuda"]): Device to run generation on ("cpu" or "cuda")

    Returns:
        None: Files are saved to the specified output directory with names {start_id}.asc to {start_id + num_of_sample - 1}.asc
             Each file contains ECG data in ASCII format with shape (5000, 8) for leads [I, II, V1, V2, V3, V4, V5, V6]
    """
    root_dir = Path(__file__).parent

    device = torch.device(runOnDevice)

    netG = Generator()
    checkpoint = torch.load(
        os.path.join(root_dir, "checkpoints/g_stat.pt"),
        map_location=device,
        weights_only=True
    )
    netG.load_state_dict(checkpoint["stat_dict"])
    netG.to(device)
    netG.eval()

    for i in tqdm(range(start_id, start_id + num_of_sample)):
        noise = torch.Tensor(1, 8, 5000).uniform_(-1, 1)
        noise = noise.to(device)
        out = netG(noise)
        out_rescaled = out * 6000
        out_rescaled = out_rescaled.int()

        out_rescaled_t = torch.t(out_rescaled.squeeze())

        # asc_file = open("{}/{}.asc".format(asc_dir, i), 'ab')
        np.savetxt("{}/{}.asc".format(out_dir, str(i)), out_rescaled_t.detach().cpu().numpy(), fmt='%i')
        # asc_file.close()


def generate_as_numpy(runOnDevice: Literal["cpu", "cuda"] = "cuda" if torch.cuda.is_available() else "cpu"):
    """Generate a single 8-lead ECG waveform using deepfakeecg model

    Args:
        runOnDevice (str): Device to run generation on ("cpu" or "cuda")

    Returns:
        numpy.ndarray: Array of shape (5000, 8) containing the ECG data
                      for leads [I, II, V1, V2, V3, V4, V5, V6]
    """
    root_dir = Path(__file__).parent
    device = torch.device(runOnDevice)

    netG = Generator()
    checkpoint = torch.load(
        os.path.join(root_dir, "checkpoints/g_stat.pt"),
        map_location=device,
        weights_only=True
    )
    netG.load_state_dict(checkpoint["stat_dict"])
    netG.to(device)
    netG.eval()

    noise = torch.Tensor(1, 8, 5000).uniform_(-1, 1)
    noise = noise.to(device)
    out = netG(noise)
    out_rescaled = out * 6000
    out_rescaled = out_rescaled.int()

    # Convert to numpy and transpose to get (5000, 8) shape
    data = out_rescaled.squeeze().t().detach().cpu().numpy()

    return data


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
def generate_NEW(numberOfSamples:   int = 1,
                 ecgType:           int = DATA_ECG8,
                 ecgLength:         int = 5000,
                 outputFormat:      int = OUTPUT_NUMPY,
                 outputFilePattern: Union[str, Path] = None,
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
   root_dir = Path(__file__).parent
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
   for i in tqdm(range(outputStartID, outputStartID + numberOfSamples)):
      # ------ Create random noise  -----------------------------------------
      # !!!!
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

      # ------ Make NumPy data ----------------------------------------------
      data = generatedECG.detach().cpu().numpy()

      # ------ Write output file --------------------------------------------
      if outputFormat in [ OUTPUT_ASC, OUTPUT_CSV ]:
        outputFileName = outputFilePattern.format(number = str(i))
        if outputFormat == OUTPUT_ASC:
           numpy.savetxt(outputFileName, data, fmt = '%i')
        elif outputFormat == OUTPUT_CSV:
           numpy.savetxt(outputFileName, data,
                         header    = 'Timestamp,LeadI,LeadII,V1,V2,V3,V4,V5,V6',
                         comments  = '',
                         delimiter = ',',
                         fmt       = '%i')

      # ------ Collect data in array ----------------------------------------
      else:
         results.append(data)

   return results
