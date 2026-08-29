def _pairs(trans_data, family):
    return list(trans_data.get(family, {}).get("pair", []))


def get_ac_pairs(trans_data):
    pairs = set(_pairs(trans_data, "AC_installed") + _pairs(trans_data, "AC"))
    pairs |= {(dst, src) for src, dst in pairs}
    return sorted(pairs)


def get_dc_pairs(trans_data):
    return sorted(set(_pairs(trans_data, "DC_installed") + _pairs(trans_data, "DC")))


def get_ac_out_neighbors(trans_data, pro):
    return [dst for src, dst in get_ac_pairs(trans_data) if src == pro]


def get_dc_out_neighbors(trans_data, pro):
    return [dst for src, dst in get_dc_pairs(trans_data) if src == pro]


def get_ac_in_neighbors(trans_data, pro):
    return [src for src, dst in get_ac_pairs(trans_data) if dst == pro]


def get_dc_in_neighbors(trans_data, pro):
    return [src for src, dst in get_dc_pairs(trans_data) if dst == pro]
