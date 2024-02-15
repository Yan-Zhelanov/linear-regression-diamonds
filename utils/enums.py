from enum import IntEnum

SetType = IntEnum('SetType', ('train', 'valid', 'test'))

PreprocessingType = IntEnum('PreprocessingType', ('normalization', 'standardization'))
