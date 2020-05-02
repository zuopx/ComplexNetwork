import src.main.util.data_format as data_format
import os
from scipy import sparse

import config
DB = config.get_DB()


def get_p_gin_from_npz(gin_mat_path):
    gin_mat = sparse.load_npz(gin_mat_path).toarray()
    return list(gin_mat.sum(axis=1) / gin_mat.shape[1])


g_name = 'DiSF_b'
base_path = os.path.join(DB, g_name)
gin_mat_base_path = os.path.join(base_path, 'gin_mat')


def main():
    for file in os.listdir(gin_mat_base_path):
        t = os.path.splitext(file)
        if t[1] == '.npz':
            print(t)
            p_gin = get_p_gin_from_npz(os.path.join(gin_mat_base_path, file))
            data_format.save_json(os.path.join(
                gin_mat_base_path, t[0] + '.json'), p_gin)


if __name__ == "__main__":
    print(g_name)
    main()