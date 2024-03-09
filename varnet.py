import ismrmrd
import os
import logging
import traceback
import numpy as np
import ctypes
import mrdhelper
import constants
from time import perf_counter
from PIL import Image
import re
import torch
import fastmri.data.transforms as T
from fastmri.models import VarNet
from typing import Callable, Optional

# Folder for debug output files
debugFolder = "/tmp/share/debug"

def process(connection, config, metadata):
    logging.info("Config: \n%s", config)

    # Metadata should be MRD formatted header, but may be a string
    # if it failed conversion earlier
    try:
        logging.info("Incoming dataset contains %d encodings", len(metadata.encoding))
        logging.info("First encoding is of type '%s', with a field of view of (%s x %s x %s)mm^3 and a matrix size of (%s x %s x %s)", 
            metadata.encoding[0].trajectory, 
            metadata.encoding[0].encodedSpace.matrixSize.x, 
            metadata.encoding[0].encodedSpace.matrixSize.y, 
            metadata.encoding[0].encodedSpace.matrixSize.z, 
            metadata.encoding[0].encodedSpace.fieldOfView_mm.x, 
            metadata.encoding[0].encodedSpace.fieldOfView_mm.y, 
            metadata.encoding[0].encodedSpace.fieldOfView_mm.z)

    except:
        logging.info("Improperly formatted metadata: \n%s", metadata)

    # Continuously parse incoming data parsed from MRD messages
    acqGroup = []
    try:
        for item in connection:
            # ----------------------------------------------------------
            # Raw k-space data messages
            # ----------------------------------------------------------
            if isinstance(item, ismrmrd.Acquisition):
                # Accumulate all imaging readouts in a group
                if (not item.is_flag_set(ismrmrd.ACQ_IS_NOISE_MEASUREMENT)              and
                    # not item.is_flag_set(ismrmrd.ACQ_IS_PARALLEL_CALIBRATION)           and
                    not item.is_flag_set(ismrmrd.ACQ_IS_PHASECORR_DATA)                 and
                    not item.is_flag_set(ismrmrd.ACQ_IS_SURFACECOILCORRECTIONSCAN_DATA) and
                    not item.is_flag_set(ismrmrd.ACQ_IS_NAVIGATION_DATA)):
                    acqGroup.append(item)

            # ----------------------------------------------------------
            # Image data messages
            # ----------------------------------------------------------
            elif isinstance(item, ismrmrd.Image):
                pass

            # ----------------------------------------------------------
            # Waveform data messages
            # ----------------------------------------------------------
            elif isinstance(item, ismrmrd.Waveform):
                pass

            elif item is None:
                break

            else:
                logging.error("Unsupported data type %s", type(item).__name__)

        # Process the group of all raw data
        if len(acqGroup) > 0:
            logging.info("Processing a group of k-space data (untriggered)")
            image = process_raw(acqGroup, connection, config, metadata)
            connection.send_image(image)
            acqGroup = []

    except Exception as e:
        logging.error(traceback.format_exc())
        connection.send_logging(constants.MRD_LOGGING_ERROR, traceback.format_exc())

    finally:
        connection.send_close()

def process_raw(group, connection, config, metadata):
    # Start timer
    tic = perf_counter()

    # Create folder, if necessary
    if not os.path.exists(debugFolder):
        os.makedirs(debugFolder)
        logging.debug("Created folder " + debugFolder + " for debug output files")

    # Device for PyTorch inference
    if torch.cuda.is_available():
        logging.info("CUDA is available to PyTorch!")
        torchDevice = 'cuda'
    else:
        logging.warning("CUDA is NOT available to PyTorch")
        torchDevice = 'cpu'

    metadata.userParameters.userParameterLong[-1].name

    lins = [acquisition.idx.kspace_encode_step_1 for acquisition in group]
    slis = [acquisition.idx.slice                for acquisition in group]

    # For interleaved acquisitions (e.g. 2D multi-slice where odd slices are acquired in one
    # concatenation and the even slices are acquired in a second concatenation), the slice index
    # is not ordered by physical slice location.  This is handled on the scanner side, but the
    # RelativeSliceNumber field can be used to determine the correct physical slice ordering
    relSliceNo = [None]*(max(slis)+1)
    for param in metadata.userParameters.userParameterLong:
        res = re.search(r'(?<=RelativeSliceNumber_)\d+$', param.name)
        if res is not None:
            relSliceNo[int(res[0])-1] = int(param.value)

    # FastMRI data is stored as with 3 variables:
    #  - kspace:         a complex64 array of shape [sli cha RO PE] containing raw data
    #  - mask:           a float32 array of shape [PE 1] containing the undersampling mask (1 for sampled, 0 otherwise)
    #  - ismrmrd_header: text serialization of the MRD header
    kspace         = np.zeros((max(slis)+1, 
                               group[0].data.shape[0], 
                               metadata.encoding[0].encodedSpace.matrixSize.x, 
                               metadata.encoding[0].encodedSpace.matrixSize.y), 
                              group[0].data.dtype)
    mask           = np.zeros((metadata.encoding[0].encodedSpace.matrixSize.y, 1), np.float32)
    ismrmrd_header = ismrmrd.xsd.ToXML(metadata)

    rawHead = [None]*(max(slis)+1)

    # Retrospectively downsample for testing using config parameters
    downsampleR = -1
    downsampleAcs = -1

    if ('parameters' in config):
        if ('downsampleR' in config['parameters']):
            downsampleR = int(config['parameters']['downsampleR'])

        if ('downsampleAcs' in config['parameters']):
            downsampleAcs = int(config['parameters']['downsampleAcs'])

    R      = -1
    offset = -1
    acs    = -1

    # if (reR is None) and (reAcs is not None):
    if (downsampleAcs != -1) and (downsampleR == -1):
        # Keep the acceleration rate and just change ACS, but we need to figure out what the acceleration rate is
        sortedLins = np.sort([acquisition.idx.kspace_encode_step_1 for acquisition in group if acquisition.idx.slice == 0])
        R = sortedLins[1]-sortedLins[0]
        offset = int(R/2)
        acs = downsampleAcs
        logging.info("Based on config, downsample data to R=%d (inferred), line offset=%d (inferred), ACS=%d", R, offset, acs)

    elif (downsampleAcs != -1) and (downsampleR != -1):
        # Change both acceleration rate and ACS
        R      = downsampleR
        offset = int(R/2)
        acs    = downsampleAcs
        logging.info("Based on config, downsample data to R=%d, ACS=%d", R, acs)

    else:
        # Figure out what the acceleration rate is
        sortedLins = np.sort([acquisition.idx.kspace_encode_step_1 for acquisition in group if acquisition.idx.slice == 0])
        inferR = sortedLins[1]-sortedLins[0]
        inferOffset = int(inferR/2)
        inferAcs = np.sum(np.diff(np.unique(sortedLins))==1) + 1
        logging.info("Based on config, not downsampling data which is inferred to be sampled at R=%d, line offset=%d, ACS=%d", inferR, inferOffset, inferAcs)

    for acq, lin, sli in zip(group, lins, slis):
        centerLin = acq.idx.user[5]

        if ((acs == -1 and R == -1 and offset == -1) or ((lin > centerLin-acs/2 and lin < centerLin+acs/2) or (lin % R) == offset)):
            kspace[sli,:,-acq.data.shape[1]:,lin] = acq.data  # Account for asymmetric echo
            mask[lin] = 1

        # Store header for each slice from line closest to center k-space
        if (rawHead[sli] is None) or (np.abs(acq.idx.kspace_encode_step_1 - centerLin) < np.abs(rawHead[sli].idx.kspace_encode_step_1 - centerLin)):
            rawHead[sli] = acq.getHead()

    np.save(debugFolder + "/" + "kspace.npy", kspace[0,...])  # Only do first slice for speed
    imMask = Image.fromarray(np.tile(mask*255, (1,int(kspace.shape[2]/2))))
    imMask.convert('RGB').save(debugFolder + "/" + "mask.png")

    device = torch.device(torchDevice)

    # Load pre-trained model weights: fastMRI End-to-End Variational Networks for Accelerated MRI Reconstruction Model
    # Sriram A et al. 2020. End-to-End Variational Networks for Accelerated MRI Reconstruction. https://doi.org/10.1007/978-3-030-59713-9_7
    # Two model files are provided, but others can be selected:
    # - varnet_brain_leaderboard_state_dict.pt
    # - varnet_knee_leaderboard_state_dict.pt

    state_dict_file = 'models/varnet_brain_leaderboard_state_dict.pt'
    if ('parameters' in config) and ('model' in config['parameters']):
        state_dict_file = 'models/' + config['parameters']['model']

    model = VarNet(num_cascades=12, pools=4, chans=18, sens_pools=4, sens_chans=8)
    model.load_state_dict(torch.load(state_dict_file, map_location=device))
    model.eval()
    model = model.to(device)

    # Prepare data
    data_transform = T.VarNetDataTransform()

    dataset = mrdSliceDataset(kspace, mask, metadata, data_transform)

    logging.info("torch.backends.openmp.is_available(): %d", torch.backends.openmp.is_available())
    logging.info("torch.get_num_threads(): %d", torch.get_num_threads())

    outputs = [None] * len(dataset)
    for sli in range(len(dataset)):
        tic2 = perf_counter()

        with torch.no_grad():
            output = model(dataset[sli].masked_kspace[None,:].to(device), dataset[sli].mask[None,:].to(device)).cpu()

        crop_size = dataset[0].crop_size
        outputs[sli] = T.center_crop(output, crop_size)[0]

        strProcessTime = "FastMRI VarNet inference time, sli %d/%d: %.2f s" % (sli, len(dataset), perf_counter()-tic2)
        logging.info(strProcessTime)
        connection.send_logging(constants.MRD_LOGGING_INFO, strProcessTime)

        if not np.any(np.array(outputs[sli])):
            logging.warning("Slice is all zeros!")

        if np.all(np.isnan(np.array(outputs[sli]))):
            logging.warning("Slice is all NaNs!")

    # img will have shape [sli RO PE]
    img = np.stack([out for out in outputs])

    logging.debug("Image data is size %s" % (img.shape,))
    np.save(debugFolder + "/" + "imgVarNet.npy", img)

    if ('parameters' in config) and ('complexoutput' in config['parameters']) and \
        ((config['parameters']['complexoutput'] == True) or (isinstance(config['parameters']['complexoutput'], str) and ('true' in config['parameters']['complexoutput'].lower()))):
        # Complex images are requested
        logging.info("Outputting complex images as requested by config")
        img = img.astype(np.complex64)
        img *= 1000
    else:
        # Determine max value (12 or 16 bit)
        BitsStored = 12
        if (mrdhelper.get_userParameterLong_value(metadata, "BitsStored") is not None):
            BitsStored = mrdhelper.get_userParameterLong_value(metadata, "BitsStored")
        maxVal = 2**BitsStored - 1

        # Normalize and convert to int16
        img = img.astype(np.float64)
        img *= maxVal/img.max()
        img = np.around(img)
        img = img.astype(np.int16)

    # Measure processing time
    strProcessTime = "Total processing time: %.2f s" % (perf_counter()-tic)
    logging.info(strProcessTime)

    # Send this as a text message back to the client
    connection.send_logging(constants.MRD_LOGGING_INFO, strProcessTime)

    # Format as ISMRMRD image data
    imagesOut = []
    for sli in range(img.shape[0]):
        # Create new MRD instance for the processed image
        # data has shape [sli RO PE], i.e. [x y], so we need to transpose.
        # from_array() should be called with 'transpose=False' to avoid warnings, and when called
        # with this option, can take input as: [cha z y x], [z y x], or [y x]
        tmpImg = ismrmrd.Image.from_array(img[sli,...].transpose(), transpose=False)

        # Set the header information
        tmpImg.setHead(mrdhelper.update_img_header_from_raw(tmpImg.getHead(), rawHead[sli]))
        tmpImg.field_of_view = (ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.x), 
                                ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.y), 
                                ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.z))

        # Use RelativeSliceNumber if available to sort by physical slice location
        # +1 so that InstanceNumbers are 1-indexed
        if all([x is not None for x in relSliceNo]):
            tmpImg.image_index = relSliceNo[sli]+1
        else:
            tmpImg.image_index = sli+1

        # Set ISMRMRD Meta Attributes
        tmpMeta = ismrmrd.Meta()
        tmpMeta['DataRole']               = 'Image'
        tmpMeta['ImageProcessingHistory'] = ['FIRE', 'PYTHON']
        tmpMeta['WindowCenter']           = '16384'
        tmpMeta['WindowWidth']            = '32768'
        tmpMeta['Keep_image_geometry']    = 1

        # Add image orientation directions to MetaAttributes if not already present
        if tmpMeta.get('ImageRowDir') is None:
            tmpMeta['ImageRowDir'] = ["{:.18f}".format(rawHead[sli].read_dir[0]), "{:.18f}".format(rawHead[sli].read_dir[1]), "{:.18f}".format(rawHead[sli].read_dir[2])]

        if tmpMeta.get('ImageColumnDir') is None:
            tmpMeta['ImageColumnDir'] = ["{:.18f}".format(rawHead[sli].phase_dir[0]), "{:.18f}".format(rawHead[sli].phase_dir[1]), "{:.18f}".format(rawHead[sli].phase_dir[2])]

        xml = tmpMeta.serialize()
        logging.debug("Image MetaAttributes: %s", xml)
        tmpImg.attribute_string = xml
        imagesOut.append(tmpImg)

    return imagesOut

class mrdSliceDataset(torch.utils.data.Dataset):
    """
    An MRD-compatible PyTorch Dataset that provides access to MR image slices based on fastmri.data.SliceDataset
    """

    def __init__(
        self,
        kspace, mask, ismrmrd_header,
        transform: Optional[Callable] = None,
    ):
        self.kspace    = kspace
        self.mask      = mask
        self.transform = transform

        self.attrs = dict({
            "padding_left":  ismrmrd_header.encoding[0].encodedSpace.matrixSize.y // 2 - ismrmrd_header.encoding[0].encodingLimits.kspace_encoding_step_1.center,
            "padding_right": ismrmrd_header.encoding[0].encodedSpace.matrixSize.y // 2 - ismrmrd_header.encoding[0].encodingLimits.kspace_encoding_step_1.center + ismrmrd_header.encoding[0].encodingLimits.kspace_encoding_step_1.maximum,
            "encoding_size": (ismrmrd_header.encoding[0].encodedSpace.matrixSize.x, ismrmrd_header.encoding[0].encodedSpace.matrixSize.y, ismrmrd_header.encoding[0].encodedSpace.matrixSize.z),
            "recon_size":    (ismrmrd_header.encoding[0].reconSpace.matrixSize.x,   ismrmrd_header.encoding[0].reconSpace.matrixSize.y,   ismrmrd_header.encoding[0].reconSpace.matrixSize.z),
        })

    def __len__(self):
        return self.kspace.shape[0]

    def __getitem__(self, i: int):
        if self.transform is None:
            sample = (self.kspace[i], self.mask, None, self.attrs, "", i)
        else:
            sample = self.transform(self.kspace[i], self.mask, None, self.attrs, "", i)

        return sample