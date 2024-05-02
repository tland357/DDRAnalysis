

class bpmList:
    def __init__(self):
        self._data = []

    def append(self, item):
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError("Item must be a tuple of length 2.")
        if not isinstance(item[0], float) or not isinstance(item[1], int):
            raise TypeError("Item must be a tuple of (float, int).")
        self._data.append(item)

    def __getitem__(self, position):
        return self._data[position]

    def __setitem__(self, position, value):
        if not isinstance(value, tuple) or len(value) != 2:
            raise TypeError("Value must be a tuple of length 2.")
        if not isinstance(value[0], float) or not isinstance(value[1], int):
            raise TypeError("Value must be a tuple of (float, int).")
        self._data[position] = value

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return repr(self._data)

class smFile:
    def __init__(self, url) -> None:
        with open(url, 'r') as file:
            lines = file.readlines()
            offsetLine = [x for x in lines if x.startswith('#OFFSET:')][0]
            self.offset = float(offsetLine.replace('#OFFSET:', '').replace('\n','').replace(';',''))
            self.bpms = bpmList()