from tabulate import tabulate
import pandas as pd
import matplotlib.pyplot as plt
    
def print_list(lis, title = ""):
    """ Print a list with a title """
    print(title)
    if len(lis) == 0:
        print(" -- Empty list.")
    else:
        for i, el in enumerate(lis, 1):
            print(f" - {i} - {el}")
 
def print_map(config_map, title = None):
    """ Print a dict with a title """
    if title:
        print(title)
        
    for i, (key, value) in enumerate(config_map.items(), 1):
        print(f"{i} - {key} : {value}")
        
def print_dataframe(data, title=None, limit=30, sort_by=None, ascending=True, show_index=True):
    
    if title is not None:
        print(title)
    pd.set_option('display.float_format', '{:.6f}'.format)
    
    # Check if data is a Series and convert to DataFrame
    if isinstance(data, pd.Series):
        
        # If the Series has a name, use it as the column name; otherwise, default to a generic column name.
        column_name = data.name if data.name is not None else 'Value'
        # Convert Series to DataFrame for uniform handling
        data = data.to_frame(name=column_name)

    if sort_by is not None and isinstance(data, pd.DataFrame):
        data = data.sort_values(by=sort_by, ascending=ascending)
    
    print(tabulate(data[:limit], headers='keys', tablefmt='simple_grid', showindex=show_index))
    print('\n')
    

def print_dict(d, title = '', avoid_keys = []):
    if title is not None:
        print(title)
    for key, value in d.items():
        if key not in avoid_keys:
                print(f" - {key}: {value}")
