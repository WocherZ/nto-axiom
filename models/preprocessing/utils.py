import os
import hashlib
from collections import defaultdict

def aggregate_names(names: list) -> set:
    """
    Create a set from the given list of names.

    Parameters:
        names (list): A list of names.

    Returns:
        set: A set containing unique names from the input list.
    """
    return set(names)

def choose_name(names: list) -> str:
    """
    A function that chooses the longest name from a list of names after filtering out names containing '№'.
    
    Parameters:
        names (list): a list of strings representing names
    
    Returns:
        str: a string which is the longest name that does not contain '№', if such a name exists; otherwise, the longest name in the original list
    """
    filtered_names = [name for name in names if '№' not in name]
    if filtered_names:
        return max(filtered_names, key=len)
    else:
        return max(names, key=len)
        
def replace_forbidden_chars(text: str) -> str:
    """
    Replaces forbidden characters in the input text with underscores and returns the modified text.

    Parameters:
        text (str): The input text to process.

    Returns:
        str: The processed text with forbidden characters replaced by underscores.
    """
    forbidden_chars = ['/', '\\', '?', '%', '*', ':', '|', '"', '<', '>', '.']
    for char in forbidden_chars:
        text = text.replace(char, '_')
    return text.rstrip()

def remove_duplicates(folder: str) -> None:
    """
    Remove duplicate files within a specified folder based on their content hash.

    Parameters:
        folder (str): The path to the folder containing the files.

    Returns:
        None
    """
    file_hashes = defaultdict(list)
    
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        
        if os.path.isdir(filepath):
            continue
        
        with open(filepath, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        
        file_hashes[file_hash].append(filepath)
    
    for files_list in file_hashes.values():
        if len(files_list) > 1:
            for file_to_remove in files_list[1:]:
                os.remove(file_to_remove)

def early_stopping(train_losses, patience=5):
    if len(train_losses) < patience + 1:
        return False
    return all(train_losses[-1] >= train_losses[-(i + 2)] for i in range(patience))