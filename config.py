class GlobalVar:

    DB = 'E:/Projects/ComplexNetwork/db'


def get_DB() -> str:
    return GlobalVar.DB
