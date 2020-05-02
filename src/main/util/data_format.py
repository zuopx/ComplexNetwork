import json

def at2edgelist(infile_path: str, outfile_path: str) -> None:
    """Conversion from adjacency table to edgelist.

    outfile_path -- file of edgelist. Create it if not existing.
    """
    at = load_json(infile_path)
    with open(outfile_path, 'w') as fw:
        for node, neighbors in enumerate(at):
            for neighbor in neighbors:
                fw.write(str(node) + '\t' + str(neighbor) + '\n')


def load_node2vec_emb(node2vec_emb_file: str) -> dict:
    node2vec_emb = {}
    fr = open(node2vec_emb_file, 'r')
    fr.readline()
    for line in fr:
        words = line.split()
        node2vec_emb[int(words[0])] = list(map(float, words[1:]))
    fr.close()
    return node2vec_emb


def load_json(file: str):
    assert file.endswith('.json')
    with open(file, 'r') as fr:
        data = json.load(fr)
    return data


def save_json(file: str, data):
    assert file.endswith('.json')
    with open(file, 'w') as fw:
        json.dump(data, fw)