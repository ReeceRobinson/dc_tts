import os
import sys
import sounddevice as sd
import soundfile as sf

# Configuration Item Candidates
record_id_prefix = "RR001"
record_id_separator = "-"
record_id_num_pad = 4
record_subdir = "wav"
processed_subdir = "wav-trim"
collection_dir = "RRSpeech-1.0"
data_dir = "data"
file_suffix = "wav"

sample_rate = 22050
sub_type = "PCM_16"
trim_samples = int(0.2 * sample_rate)  # 0.2 seconds to remove mouse click when stopping the recording.

data_dir_structure = os.path.join(data_dir, collection_dir, record_subdir)
processed_dir_structure = os.path.join(data_dir, collection_dir, processed_subdir)


def getRecordingFilename(current_sentence_index):
    name = getFileName(current_sentence_index)
    return os.path.join(data_dir_structure, name)


def getProcessedFilename(current_sentence_index):
    name = getFileName(current_sentence_index)
    return os.path.join(processed_dir_structure, name)


def getFileName(current_sentence_index):
    index_format = '{2:0' + str(record_id_num_pad) + '}'
    name_format = '{0}{1}' + index_format
    name = name_format.format(record_id_prefix, record_id_separator, current_sentence_index)
    return name


def ensureDir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)


def getFiles(suffix):
    items = os.listdir(data_dir_structure)
    item_list = list(filter(lambda x: x.endswith(suffix), items))
    return item_list

files = getFiles(file_suffix)
ensureDir(processed_dir_structure)

for f in files:
    filename = os.path.join(data_dir_structure, f)
    out_filename = os.path.join(processed_dir_structure, f)
    try:
        data, fs = sf.read(filename, dtype='float32')
        size = len(data)
        with sf.SoundFile(out_filename, mode='x', samplerate=sample_rate, channels=1, subtype=sub_type) as file:
            file.write(data[0:size-trim_samples])

    except Exception as e:
        exit(type(e).__name__ + ': ' + str(e))
