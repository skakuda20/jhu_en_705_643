import os
import patoolib

# src: path for hmdb51_org.rar
# dest_dir: directory to store all extracted folders of HMDB51
def unrar_dataset(src, dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir, exist_ok = False)
    patoolib.extract_archive(src, outdir=dest_dir)
    for rar_f in os.listdir(dest_dir):
        if rar_f.endswith('.rar'):
            rar_path = os.path.join(dest_dir, rar_f)
            patoolib.extract_archive(rar_path, outdir=dest_dir)
            os.remove(rar_path)

if __name__=="__main__":
    src_path = '/home/vince/datasets/src_data/hmdb51_org.rar'
    dest_path = '/home/vince/datasets/raw_data/HMDB51'
    unrar_dataset(src_path, dest_path)
