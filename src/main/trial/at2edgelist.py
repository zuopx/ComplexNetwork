import os

import src.main.util.data_format as data_format

import config
DB = config.get_DB()


g_name = 'DiSF_b'

base_path = os.path.join(DB, g_name)

at_path = os.path.join(base_path, 'at.json')
edgelist_path = os.path.join(base_path, 'edgelist')
emb_path = os.path.join(base_path, 'node2vec.emb')
# edgelist_path = os.path.join(base_path, 'edgelist.json')


def main():
    data_format.at2edgelist(at_path, edgelist_path)
    # edgelist = data_format.load_node2vec_emb(edgelist_path)
    # data_format.save_json()


def main2():
    emb_dict = data_format.load_node2vec_emb(emb_path)
    data_format.save_json(os.path.join(
        base_path, 'node2vec.emb.json'), emb_dict)


if __name__ == "__main__":
    main2()
