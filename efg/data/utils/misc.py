def _dict_select(dict_, inds):
    for k, v in dict_.items():
        if isinstance(v, dict):
            _dict_select(v, inds)
        else:
            try:
                dict_[k] = v[inds]
            except IndexError:
                dict_[k] = v[inds[len(v)]]
