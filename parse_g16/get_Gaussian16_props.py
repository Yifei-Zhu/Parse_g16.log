import numpy as np
import pandas as pd
from src.sub_code_get_GS_opt_props import main_import as get_GS_props
from src.sub_code_get_ES_opt_props import main_import as get_ES_props


def get_GS_props_from_log(input_file, save_file,state,work_type):
    log_file=f'{input_file}'
    prop_file=f'{save_file}'
    state_to_function = {
                        'GS': get_GS_props,
                        'ES': get_ES_props
                        }
    state_to_function[state](log_file, prop_file, work_type)
    print('Done!')

if __name__ == '__main__':

    print('Done!')
